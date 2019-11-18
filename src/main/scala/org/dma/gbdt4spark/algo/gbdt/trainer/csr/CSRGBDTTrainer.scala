package org.dma.gbdt4spark.algo.gbdt.trainer.csr

import java.util.concurrent.Executors

import org.apache.spark.ml.linalg.Vector
import org.dma.gbdt4spark.algo.gbdt.helper.SplitFinder
import org.dma.gbdt4spark.algo.gbdt.metadata.{FeatureInfo, InstanceInfo}
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTSplit, GBTTree}
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.{Maths, RangeBitSet}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import java.{util => ju}

import org.dma.gbdt4spark.algo.gbdt.histogram.{BinaryGradPair, GradPair, MultiGradPair}
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.loss.Loss
import org.dma.gbdt4spark.objective.metric.EvalMetric
import org.dma.gbdt4spark.objective.metric.EvalMetric.Kind
import org.dma.gbdt4spark.tree.split.SplitEntry

class CSRGBDTTrainer(val workerId: Int, val param: GBDTParam,
                     @transient private[gbdt] val featureInfo: FeatureInfo,
                     @transient private[gbdt] val trainData: CSRDataset,
                     @transient private[gbdt] val  labels: Array[Float],
                     @transient private[gbdt] val validData: Array[Vector],
                     @transient private[gbdt] val validLabels: Array[Float]) extends Serializable {
  private[gbdt] val forest = ArrayBuffer[GBTTree]()

  @transient private[gbdt] val validPreds = {
    if (param.numClass == 2)
      new Array[Float](validData.length)
    else
      new Array[Float](validData.length * param.numClass)
  }

  private[gbdt] val instanceInfo = CSRInstanceInfo(param, labels.length)
  private[gbdt] val numFeatUsed = Math.round(featureInfo.numFeature * param.featSampleRatio)
  private[gbdt] val isFeatUsed = {
    if (numFeatUsed == featureInfo.numFeature)
      (0 until numFeatUsed).map(fid => featureInfo.getNumBin(fid) > 0).toArray
    else
      new Array[Boolean](numFeatUsed)
  }

  private[gbdt] val activeNodes = ArrayBuffer[Int]()
  @transient private[gbdt] val histManager = CSRHistManager(param, featureInfo)
  @transient private[gbdt] val splitFinder = SplitFinder(param, featureInfo)
  @transient private[gbdt] val threadPool = if (param.numThread > 1)
    Executors.newFixedThreadPool(param.numThread) else null

  @transient private[gbdt] val buildHistTime = new Array[Long](Maths.pow(2, param.maxDepth) - 1)
  @transient private[gbdt] val findSplitTime = new Array[Long](Maths.pow(2, param.maxDepth) - 1)
  @transient private[gbdt] val getSplitResultTime = new Array[Long](Maths.pow(2, param.maxDepth) - 1)
  @transient private[gbdt] val splitNodeTime = new Array[Long](Maths.pow(2, param.maxDepth) - 1)

  def timing[A](f: => A)(t: Long => Any): A = {
    val t0 = System.currentTimeMillis()
    val res = f
    t(System.currentTimeMillis() - t0)
    res
  }

  def reportTime(): String = {
    val sb = new StringBuilder
    for (depth <- 0 until param.maxDepth) {
      val from = Maths.pow(2, depth) - 1
      val until = Maths.pow(2, depth + 1) - 1
      if (from < Maths.pow(2, param.maxDepth) - 1) {
        sb.append(s"Layer${depth + 1}:\n")
        sb.append(s"|buildHistTime: [${buildHistTime.slice(from, until).mkString(", ")}], " +
          s"sum[${buildHistTime.slice(from, until).sum}]\n")
        sb.append(s"|findSplitTime: [${findSplitTime.slice(from, until).mkString(", ")}], " +
          s"sum[${findSplitTime.slice(from, until).sum}]\n")
        sb.append(s"|getSplitResultTime: [${getSplitResultTime.slice(from, until).mkString(", ")}], " +
          s"sum[${getSplitResultTime.slice(from, until).sum}]\n")
        sb.append(s"|splitNodeTime: [${splitNodeTime.slice(from, until).mkString(", ")}], " +
          s"sum[${splitNodeTime.slice(from, until).sum}]\n")
      }
    }
    val res = sb.toString()
    println(res)
    for (i <- buildHistTime.indices) {
      buildHistTime(i) = 0
      findSplitTime(i) = 0
      getSplitResultTime(i) = 0
      splitNodeTime(i) = 0
    }
    res
  }

  def createNewTree(): Unit = {
    // 1. create new tree
    val tree = new GBTTree(param)
    this.forest += tree
    // 2. sample features
    if (numFeatUsed != featureInfo.numFeature) {
      ju.Arrays.fill(isFeatUsed, false)
      for (_ <- 0 until numFeatUsed) {
        val rand = Random.nextInt(featureInfo.numFeature)
        isFeatUsed(rand) = featureInfo.getNumBin(rand) > 0
      }
      histManager.reset(isFeatUsed)
    } else if (forest.length == 1) {
      histManager.reset(isFeatUsed)
    }
    // 3. reset position info
    instanceInfo.resetPosInfo()
    // 4. calc grads
    val loss = getLoss
    val sumGradPair = instanceInfo.calcGradPairs(labels, loss, param, threadPool)
    tree.getRoot.setSumGradPair(sumGradPair)
    histManager.setGradPair(0, sumGradPair)
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

  def getSplitResults(splits: Seq[(Int, Int, Int, GBTSplit)]): Seq[(Int, RangeBitSet)] = {
    val tree = forest.last
    splits.flatMap {
      case (nid, ownerId, fidInWorker, split) =>
        tree.getNode(nid).setSplitEntry(split.getSplitEntry)
        histManager.setGradPair(2 * nid + 1, split.getLeftGradPair)
        histManager.setGradPair(2 * nid + 2, split.getRightGradPair)
        if (ownerId == this.workerId)
          Iterator((nid, getSplitResult(nid, fidInWorker, split.getSplitEntry)))
        else
          Iterator.empty
    }
  }

  def splitNodes(splitResults: Seq[(Int, RangeBitSet)]): Boolean = {
    splitResults.foreach {
      case (nid, result) =>
        splitNode(nid, result, (histManager.getGradPair(2 * nid + 1),
          histManager.getGradPair(2 * nid + 2)))
        if (2 * nid + 1 < Maths.pow(2, param.maxDepth) - 1) {
          activeNodes += 2 * nid + 1
          activeNodes += 2 * nid + 2
        } else {
          histManager.removeNodeHist(nid)
        }
    }
    activeNodes.nonEmpty
  }

  def buildHistAndFindSplit(nids: Seq[Int]): Seq[(Int, GBTSplit)] = {
    val canSplits = nids.map(canSplitNode)

    val buildStart = System.currentTimeMillis()
    var cur = 0
    val toBuild = ArrayBuffer[Int]()
    val toSubtract = ArrayBuffer[Boolean]()
    while (cur < nids.length) {
      val nid = nids(cur)
      val sibNid = Maths.sibling(nid)
      if (cur + 1 < nids.length && nids(cur + 1) == sibNid) {
        if (canSplits(cur) || canSplits(cur + 1)) {
          val curSize = instanceInfo.getNodeSize(nid)
          val sibSize = instanceInfo.getNodeSize(sibNid)
          if (curSize < sibSize) {
            toBuild += nid
            toSubtract += canSplits(cur + 1)
          } else {
            toBuild += sibNid
            toSubtract += canSplits(cur)
          }
        } else {
          histManager.removeNodeHist(Maths.parent(nid))
        }
        cur += 2
      } else {
        if (canSplits(cur)) {
          toBuild += nid
          toSubtract += false
        }
        cur += 1
      }
    }
    timing {
      if (toBuild.head == 0) {
        histManager.buildHistForRoot(trainData, instanceInfo, threadPool)
      } else {
        histManager.buildHistForNodes(toBuild, trainData, instanceInfo, toSubtract, threadPool)
      }
    } {t => buildHistTime(nids.min) = t}
    println(s"Build histograms cost ${System.currentTimeMillis() - buildStart} ms")

    val findStart = System.currentTimeMillis()
    val res = (nids, canSplits).zipped.map {
      case (nid, true) =>
        val hist = histManager.getNodeHist(nid)
        val sumGradPair = histManager.getGradPair(nid)
        val nodeGain = forest.last.getNode(nid).calcGain(param)
        val split = timing {
          splitFinder.findBestSplit(hist, sumGradPair, nodeGain)
        } {t => findSplitTime(nid) = t}
        (nid, if (split.isValid(param.minSplitGain)) split else new GBTSplit())
      case (nid, false) =>
        setAsLeaf(nid)
        (nid, new GBTSplit())
    }.filter(_._2.isValid(param.minSplitGain))
    println(s"Find splits cost ${System.currentTimeMillis() - findStart} ms")
    res
  }

  def getSplitResult(nid: Int, fidInWorker: Int, splitEntry: SplitEntry): RangeBitSet = {
    require(!splitEntry.isEmpty && splitEntry.getGain > param.minSplitGain)
    val splits = featureInfo.getSplits(fidInWorker)
    timing(instanceInfo.getSplitResult2(nid, fidInWorker, splitEntry, splits, trainData, threadPool)) {
      t => getSplitResultTime(nid) = t
    }
  }

  def splitNode(nid: Int, splitResult: RangeBitSet,
                childrenGradPairs: (GradPair, GradPair)): Unit = {
    timing {
      instanceInfo.updatePos(nid, splitResult)
      val tree = forest.last
      val node = tree.getNode(nid)
      val leftChild = new GBTNode(2 * nid + 1, node, param.numClass)
      val rightChild = new GBTNode(2 * nid + 2, node, param.numClass)
      node.setLeftChild(leftChild)
      node.setRightChild(rightChild)
      tree.setNode(2 * nid + 1, leftChild)
      tree.setNode(2 * nid + 2, rightChild)
      leftChild.setSumGradPair(childrenGradPairs._1)
      rightChild.setSumGradPair(childrenGradPairs._2)
    } {t => splitNodeTime(nid) = t}
  }

  def canSplitNode(nid: Int): Boolean = {
    if (instanceInfo.getNodeSize(nid) > param.minNodeInstance) {
      if (param.numClass == 2) {
        val sumGradPair = histManager.getGradPair(nid).asInstanceOf[BinaryGradPair]
        param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
      } else {
        val sumGradPair = histManager.getGradPair(nid).asInstanceOf[MultiGradPair]
        param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
      }
    } else {
      false
    }
  }

  def setAsLeaf(nid: Int): Unit = setAsLeaf(nid, forest.last.getNode(nid))

  def setAsLeaf(nid: Int, node: GBTNode): Unit = {
    node.chgToLeaf()
    // TODO: update predictions of all training instance together
    if (param.numClass == 2) {
      val weight = node.calcWeight(param)
      instanceInfo.updatePreds(nid, weight, param.learningRate)
    } else {
      val weights = node.calcWeights(param)
      instanceInfo.updatePreds(nid, weights, param.learningRate)
    }
    if (nid < Maths.pow(2, param.maxDepth) - 1)  // node not on the last level
      histManager.removeNodeHist(nid)
  }


  def finishTree(): Unit = {
    forest.last.getNodes.asScala.foreach {
      case (nid, node) =>
        if (node.getSplitEntry == null && !node.isLeaf)
          setAsLeaf(nid, node)
    }
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

    val slice = Maths.avgSlice(labels.length, param.numWorker, workerId)
    val evalMetrics = getEvalMetrics
    val metrics = evalMetrics.map(evalMetric => {
      val kind = evalMetric.getKind
      val trainSum = evalMetric.sum(instanceInfo.predictions, labels, slice._1, slice._2)
      val trainMetric = kind match {
        case Kind.AUC => evalMetric.avg(trainSum, 1)
        case _ => evalMetric.avg(trainSum, slice._2 - slice._1)
      }
      val validSum = evalMetric.sum(validPreds, validLabels)
      val validMetric = kind match {
        case Kind.AUC => evalMetric.avg(validSum, 1)
        case _ => evalMetric.avg(validSum, validLabels.length)
      }
      (kind, trainSum, trainMetric, validSum, validMetric)
    })

    val evalTrainMsg = metrics.map(metric => s"${metric._1}[${metric._3}]").mkString(", ")
    println(s"Evaluation on train data after ${forest.size} tree(s): $evalTrainMsg")
    val evalValidMsg = metrics.map(metric => s"${metric._1}[${metric._5}]").mkString(", ")
    println(s"Evaluation on valid data after ${forest.size} tree(s): $evalValidMsg")
    metrics.map(metric => (metric._1, metric._2, metric._4))
  }

  def finalizeModel(): Seq[GBTTree] = {
    println(s"Worker[$workerId] finalizing...")
    if (threadPool != null) threadPool.shutdown()
    forest
  }

  def getLoss: Loss = ObjectiveFactory.getLoss(param.lossFunc)

  def getEvalMetrics: Array[EvalMetric] = ObjectiveFactory.getEvalMetricsOrDefault(param.evalMetrics, getLoss)



}







