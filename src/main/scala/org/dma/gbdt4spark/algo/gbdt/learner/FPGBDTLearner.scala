package org.dma.gbdt4spark.algo.gbdt.learner

import org.apache.spark.ml.linalg.Vector
import org.dma.gbdt4spark.algo.gbdt.histogram.{GradPair, HistBuilder, Histogram, SplitFinder}
import org.dma.gbdt4spark.algo.gbdt.metadata.{DataInfo, FeatureInfo}
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTSplit, GBTTree}
import org.dma.gbdt4spark.data.{FeatureRow, InstanceRow}
import org.dma.gbdt4spark.exception.GBDTException
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.metric.EvalMetric
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{EvenPartitioner, Maths, RangeBitSet}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import java.{util => ju}

object FPGBDTLearner {
  def ensureLabel(labels: Array[Float], numLabel: Int): Unit = {
    var min = Integer.MAX_VALUE
    var max = Integer.MIN_VALUE
    for (label <- labels) {
      val trueLabel = label.toInt
      min = Math.min(min, trueLabel)
      max = Math.max(max, trueLabel)
      if (label < 0 || label > numLabel)
        throw new GBDTException("Incorrect label: " + trueLabel)
    }
    if (max - min >= numLabel) {
      throw new GBDTException(s"Invalid range for labels: [$min, $max]")
    } else if (max == numLabel) {
      println(s"Change range of labels from [1, $numLabel] to [0, ${numLabel - 1}]")
      for (i <- labels.indices)
        labels(i) -= 1
    }
  }

  def retranspose(dataFP: Array[Option[FeatureRow]], numData: Int, featLo: Int): Array[InstanceRow] = {
    val nnzs = new Array[Int](numData)
    dataFP.foreach {
      case Some(featRow) => featRow.indices.foreach(nnzs(_) += 1)
      case None =>
    }

    val insRows = new Array[InstanceRow](numData)
    for (i <- 0 until numData) {
      if (nnzs(i) > 0) {
        insRows(i) = InstanceRow(new Array[Int](nnzs(i)), new Array[Int](nnzs(i)))
        nnzs(i) = 0
      }
    }
    for (i <- dataFP.indices) {
      if (dataFP(i).isDefined) {
        val featId = featLo + i
        val featRow = dataFP(i).get
        val indices = featRow.indices
        val bins = featRow.bins
        for (j <- indices.indices) {
          val insId = indices(j)
          val binId = bins(j)
          insRows(insId).indices(nnzs(insId)) = featId
          insRows(insId).bins(nnzs(insId)) = binId
          nnzs(insId) += 1
        }
      }
    }

    insRows
  }
}

class FPGBDTLearner(val workerId: Int, val param: GBDTParam, _featureInfo: FeatureInfo,
                    _trainData: Array[InstanceRow], _labels: Array[Float],
                    _validData: Array[Vector], _validLabel: Array[Float]) {
  @transient private[learner] val forest = ArrayBuffer[GBTTree]()

  @transient private val trainData: Array[InstanceRow] = _trainData
  @transient private val labels: Array[Float] = _labels
  @transient private val validData: Array[Vector] = _validData
  @transient private val validLabels: Array[Float] = _validLabel
  @transient private val validPreds = new Array[Float](validData.length)

  private[learner] val (featLo, featHi) = {
    val featureEdges = new EvenPartitioner(param.numFeature, param.numWorker).partitionEdges()
    (featureEdges(workerId), featureEdges(workerId + 1))
  }
  private[learner] val numFeatUsed = Math.round((featHi - featLo) * param.featSampleRatio)
  private[learner] val isFeatUsed =
    if (numFeatUsed == featHi - featLo) {
      (featLo until featHi).map(fid => _featureInfo.getNumBin(fid) > 0).toArray
    } else {
      new Array[Boolean](featHi - featLo)
    }
  private[learner] val featureInfo: FeatureInfo = _featureInfo
  private[learner] val dataInfo = DataInfo(param, labels.length)

  private[learner] val loss = ObjectiveFactory.getLoss(param.lossFunc)
  private[learner] val evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(param.evalMetrics, loss)

  private[learner] val histBuilder = new HistBuilder(param)
  private[learner] val splitFinder = new SplitFinder(param)

  private[learner] val gatherBestSplit: (Int, Int, Int, GBTSplit) => GBTSplit = null
  private[learner] val broadcastSplitResult: (Int, Int, Int, RangeBitSet) => RangeBitSet = null
  private[learner] val askForSplitResult: (Int, Int, Int) => RangeBitSet = null

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
    dataInfo.calcGradPairs(0, labels, loss, param)
    tree.getRoot.setSumGradPair(dataInfo.sumGradPair(0))
  }

  def buildHistograms(nid: Int): Array[Histogram] = {
    histBuilder.buildHistogramsFP(isFeatUsed, featLo, trainData, featureInfo, dataInfo,
      nid, forest.last.getNode(nid).getSumGradPair)
  }

  def histSubtraction(parHist: Array[Histogram], sibHist: Array[Histogram]): Unit = {
    for (i <- isFeatUsed.indices) {
      if (isFeatUsed(i)) {
        parHist(i).subtractBy(sibHist(i))
      }
    }
  }

  def findLocalBestSplit(nid: Int, histograms: Array[Histogram]): GBTSplit = {
    val node = forest.last.getNode(nid)
    val sumGradPair = node.getSumGradPair
    val nodeGain = node.calcGain(param)
    splitFinder.findBestSplitFP(featLo, histograms, featureInfo, sumGradPair, nodeGain)
  }

  def findGlobalBestSplit(nid: Int, histograms: Array[Histogram]): GBTSplit = {
    val localBestSplit = findLocalBestSplit(nid, histograms)
    val treeId = 0
    gatherBestSplit(treeId, nid, workerId, localBestSplit)
  }

  def splitNode(nid: Int, sumGradPair: GradPair, split: GBTSplit): Unit = {
    val splitEntry = split.getSplitEntry
    if (!splitEntry.isEmpty && splitEntry.getGain > param.minSplitGain) {
      val splitResult = getSplitResult(nid, splitEntry) match {
        case Some(result) => broadcastSplitResult(0, 0, 0, result)
        case None => askForSplitResult(0, 0, 0)
      }
      splitNode(nid, splitResult)
    } else {
      setAsLeaf(nid)
    }
  }

  def getSplitResult(nid: Int, splitEntry: SplitEntry): Option[RangeBitSet] = {
    require(!splitEntry.isEmpty && splitEntry.getGain > param.minSplitGain)
    forest.last.getNode(nid).setSplitEntry(splitEntry)
    val splitFid = splitEntry.getFid
    if (featLo <= splitFid && splitFid < featHi) {
      val splits = featureInfo.getSplits(splitFid)
      Option(dataInfo.getSplitResult(nid, splitEntry, splits, trainData))
    } else {
      Option.empty
    }
  }

  def splitNode(nid: Int, splitResult: RangeBitSet): Unit = {
    val t0 = System.currentTimeMillis()
    dataInfo.updatePos(nid, splitResult)
    val t1 = System.currentTimeMillis()
    val tree = forest.last
    val node = tree.getNode(nid)
    val leftChild = new GBTNode(2 * nid + 1, node, param.numClass)
    val rightChild = new GBTNode(2 * nid + 2, node, param.numClass)
    node.setLeftChild(leftChild)
    node.setRightChild(rightChild)
    tree.setNode(2 * nid + 1, leftChild)
    tree.setNode(2 * nid + 2, rightChild)
    val t2 = System.currentTimeMillis()
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
    val t3 = System.currentTimeMillis()
    println(s"Split node[$nid] cost ${t3 - t0} ms, including " +
      s"updatePos[${t1 - t0}], setNodes[${t2 - t1}], setGradPair[${t3 - t2}]")
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
  }

  def evaluate(): Array[(EvalMetric.Kind, Double, Double)] = {
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

}
