package org.dma.gbdt4spark.algo.gbdt.learner

import org.apache.spark.ml.linalg.Vector
import org.dma.gbdt4spark.algo.gbdt.histogram._
import org.dma.gbdt4spark.algo.gbdt.metadata.{DataInfo, FeatureInfo}
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTSplit, GBTTree}
import org.dma.gbdt4spark.data.InstanceRow
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.metric.EvalMetric
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{EvenPartitioner, Maths, RangeBitSet}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import java.{util => ju}

class FPGBDTLearner(val learnerId: Int, val param: GBDTParam, _featureInfo: FeatureInfo,
                    _trainData: Array[InstanceRow], _labels: Array[Float],
                    _validData: Array[Vector], _validLabel: Array[Float]) {
  @transient private[learner] val forest = ArrayBuffer[GBTTree]()

  @transient private val trainData: Array[InstanceRow] = _trainData
  @transient private val labels: Array[Float] = _labels
  @transient private val validData: Array[Vector] = _validData
  @transient private val validLabels: Array[Float] = _validLabel
  @transient private val validPreds = {
    if (param.numClass == 2)
      new Array[Float](validData.length)
    else
      new Array[Float](validData.length * param.numClass)
  }

  private[learner] val (featLo, featHi) = {
    val featureEdges = new EvenPartitioner(param.numFeature, param.numWorker).partitionEdges()
    (featureEdges(learnerId), featureEdges(learnerId + 1))
  }
  private[learner] val numFeatUsed = Math.round((featHi - featLo) * param.featSampleRatio)
  private[learner] val isFeatUsed = {
    if (numFeatUsed == featHi - featLo)
      (featLo until featHi).map(fid => _featureInfo.getNumBin(fid) > 0).toArray
    else
      new Array[Boolean](featHi - featLo)
  }
  private[learner] val featureInfo: FeatureInfo = _featureInfo
  private[learner] val dataInfo = DataInfo(param, labels.length)

  private[learner] val loss = ObjectiveFactory.getLoss(param.lossFunc)
  private[learner] val evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(param.evalMetrics, loss)

  // histograms and global best splits, one for each internal tree node
  private[learner] val storedHists = new Array[Array[Histogram]](Maths.pow(2, param.maxDepth) - 1)
  private[learner] val bestSplits = new Array[GBTSplit](Maths.pow(2, param.maxDepth) - 1)
  private[learner] val histBuilder = new HistBuilder(param)
  private[learner] val splitFinder = new SplitFinder(param)

  private[learner] val activeNodes = ArrayBuffer[Int]()

  def createNewTree(): Unit = {
    // 1. create new tree
    val tree = new GBTTree(param)
    this.forest += tree
    // 2. sample features
    if (numFeatUsed != featHi - featLo) {
      ju.Arrays.fill(isFeatUsed, false)
      for (_ <- 0 until numFeatUsed) {
        val rand = Random.nextInt(featHi - featLo)
        isFeatUsed(rand) = featureInfo.getNumBin(featLo + rand) > 0
      }
    }
    // 3. reset position info
    dataInfo.resetPosInfo()
    // 4. calc grads
    val sumGradPair = dataInfo.calcGradPairs(0, labels, loss, param)
    tree.getRoot.setSumGradPair(sumGradPair)
    // 5. set root status
    activeNodes += 0
  }

  def findSplits(): Seq[(Int, GBTSplit)] = {
    val res = if (activeNodes.nonEmpty) {
      buildHistAndFindSplit(activeNodes)
    } else {
      Seq.empty
    }
    activeNodes.clear()
    res
  }

  def getSplitResults(splits: Seq[(Int, GBTSplit)]): Seq[(Int, RangeBitSet)] = {
    val tree = forest.last
    splits.map {
      case (nid, split) =>
        tree.getNode(nid).setSplitEntry(split.getSplitEntry)
        bestSplits(nid) = split
        (nid, getSplitResult(nid, split.getSplitEntry))
    }.filter(_._2 != null)
  }

  def splitNodes(splitResults: Seq[(Int, RangeBitSet)]): Boolean = {
    splitResults.foreach {
      case (nid, result) =>
        splitNode(nid, result, bestSplits(nid))
        if (2 * nid + 1 < storedHists.length) {
          activeNodes += 2 * nid + 1
          activeNodes += 2 * nid + 2
        }
    }
    activeNodes.nonEmpty
  }

  def buildHistAndFindSplit(nids: Seq[Int]): Seq[(Int, GBTSplit)] = {
    val nodes = nids.map(forest.last.getNode)
    val sumGradPairs = nodes.map(_.getSumGradPair)
    val canSplits = nodes.map(canSplitNode)

    val buildStart = System.currentTimeMillis()
    var cur = 0
    while (cur < nids.length) {
      val nid = nids(cur)
      val sibNid = Maths.sibling(nid)
      if (cur + 1 < nids.length && nids(cur + 1) == sibNid) {
        if (canSplits(cur) || canSplits(cur + 1)) {
          val curSize = dataInfo.getNodeSize(nid)
          val sibSize = dataInfo.getNodeSize(sibNid)
          val parNid = Maths.parent(nid)
          val parHist = storedHists(parNid)
          if (curSize < sibSize) {
            storedHists(nid) = histBuilder.buildHistogramsFP(
              isFeatUsed, featLo, trainData, featureInfo, dataInfo,
              nid, sumGradPairs(cur)
            )
            storedHists(sibNid) = histBuilder.histSubtraction(
              parHist, storedHists(nid), true
            )
          } else {
            storedHists(sibNid) = histBuilder.buildHistogramsFP(
              isFeatUsed, featLo, trainData, featureInfo, dataInfo,
              sibNid, sumGradPairs(cur + 1)
            )
            storedHists(nid) = histBuilder.histSubtraction(
              parHist, storedHists(sibNid), true
            )
          }
          storedHists(parNid) = null
        }
        cur += 2
      } else {
        if (canSplits(cur)) {
          storedHists(nid) = histBuilder.buildHistogramsFP(
            isFeatUsed, featLo, trainData, featureInfo, dataInfo,
            nid, sumGradPairs(cur)
          )
        }
        cur += 1
      }
    }
    println(s"Build histograms cost ${System.currentTimeMillis() - buildStart} ms")

    val findStart = System.currentTimeMillis()
    val res = canSplits.zipWithIndex.map {
      case (canSplit, i) =>
        val nid = nids(i)
        if (canSplit) {
          val node = nodes(i)
          val hist = storedHists(nid)
          val sumGradPair = sumGradPairs(i)
          val nodeGain = node.calcGain(param)
          val split = splitFinder.findBestSplitFP(featLo, hist,
            featureInfo, sumGradPair, nodeGain)
          (nid, split)
        } else {
          (nid, new GBTSplit)
        }
    }.filter(_._2.isValid(param.minSplitGain))
    println(s"Find splits cost ${System.currentTimeMillis() - findStart} ms")
    res
  }

  def getSplitResult(nid: Int, splitEntry: SplitEntry): RangeBitSet = {
    require(!splitEntry.isEmpty && splitEntry.getGain > param.minSplitGain)
    //forest.last.getNode(nid).setSplitEntry(splitEntry)
    val splitFid = splitEntry.getFid
    if (featLo <= splitFid && splitFid < featHi) {
      val splits = featureInfo.getSplits(splitFid)
      dataInfo.getSplitResult(nid, splitEntry, splits, trainData)
    } else {
      null
    }
  }

  def splitNode(nid: Int, splitResult: RangeBitSet, split: GBTSplit = null): Unit = {
    dataInfo.updatePos(nid, splitResult)
    val tree = forest.last
    val node = tree.getNode(nid)
    val leftChild = new GBTNode(2 * nid + 1, node, param.numClass)
    val rightChild = new GBTNode(2 * nid + 2, node, param.numClass)
    node.setLeftChild(leftChild)
    node.setRightChild(rightChild)
    tree.setNode(2 * nid + 1, leftChild)
    tree.setNode(2 * nid + 2, rightChild)
    if (split == null) {
      val leftSize = dataInfo.getNodeSize(2 * nid + 1)
      val rightSize = dataInfo.getNodeSize(2 * nid + 2)
      if (leftSize < rightSize) {
        val leftSumGradPair = dataInfo.sumGradPair(2 * nid + 1)
        val rightSumGradPair = node.getSumGradPair.subtract(leftSumGradPair)
        leftChild.setSumGradPair(leftSumGradPair)
        rightChild.setSumGradPair(rightSumGradPair)
      } else {
        val rightSumGradPair = dataInfo.sumGradPair(2 * nid + 2)
        val leftSumGradPair = node.getSumGradPair.subtract(rightSumGradPair)
        leftChild.setSumGradPair(leftSumGradPair)
        rightChild.setSumGradPair(rightSumGradPair)
      }
    } else {
      leftChild.setSumGradPair(split.getLeftGradPair)
      rightChild.setSumGradPair(split.getRightGradPair)
    }
  }

  def canSplitNode(node: GBTNode): Boolean = {
    if (dataInfo.getNodeSize(node.getNid) > param.minNodeInstance) {
      if (param.numClass == 2) {
        val sumGradPair = node.getSumGradPair.asInstanceOf[BinaryGradPair]
        param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
      } else {
        val sumGradPair = node.getSumGradPair.asInstanceOf[MultiGradPair]
        param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
      }
    } else {
      false
    }
  }

  def setAsLeaf(nid: Int): Unit = setAsLeaf(nid, forest.last.getNode(nid))

  def setAsLeaf(nid: Int, node: GBTNode): Unit = {
    node.chgToLeaf()
    if (param.numClass == 2) {
      val weight = node.calcWeight(param)
      dataInfo.updatePreds(nid, weight, param.learningRate)
    } else {
      val weights = node.calcWeights(param)
      dataInfo.updatePreds(nid, weights, param.learningRate)
    }
  }

  def finishTree(): Unit = {
    forest.last.getNodes.asScala.foreach {
      case (nid, node) =>
        if (node.getSplitEntry == null && !node.isLeaf)
          setAsLeaf(nid, node)
    }
    for (i <- storedHists.indices)
      storedHists(i) = null
  }

  def evaluate(): Seq[(EvalMetric.Kind, Double, Double)] = {
    for (i <- validData.indices) {
      var node = forest.last.getRoot
      while (!node.isLeaf) {
        if (node.getSplitEntry.flowTo(validData(i)) == 0)
          node = node.getLeftChild.asInstanceOf[GBTNode]
        else
          node = node.getRightChild.asInstanceOf[GBTNode]
      }
      if (param.numClass == 2) {
        validPreds(i) += node.getWeight * param.learningRate
      } else {
        val weights = node.getWeights
        for (k <- 0 until param.numClass)
          validPreds(i * param.numClass + k) += weights(k) * param.learningRate
      }
    }

    val metrics = evalMetrics.map(evalMetric =>
      (evalMetric.getKind, evalMetric.eval(dataInfo.predictions, labels),
        evalMetric.eval(validPreds, validLabels))
    )

    val evalTrainMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
    println(s"Evaluation on train data after ${forest.size} tree(s): $evalTrainMsg")
    val evalValidMsg = metrics.map(metric => s"${metric._1}[${metric._3}]").mkString(", ")
    println(s"Evaluation on valid data after ${forest.size} tree(s): $evalValidMsg")
    metrics
  }

  def finalizeModel(): Seq[GBTTree] = {
    histBuilder.shutdown()
    forest
  }

}
