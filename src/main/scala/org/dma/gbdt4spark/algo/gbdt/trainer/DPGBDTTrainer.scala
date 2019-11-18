package org.dma.gbdt4spark.algo.gbdt.trainer

import java.util.concurrent.Executors

import org.apache.spark.ml.linalg.Vector
import org.dma.gbdt4spark.algo.gbdt.dataset.Dataset
import org.dma.gbdt4spark.algo.gbdt.helper.{HistManager, SplitFinder}
import org.dma.gbdt4spark.algo.gbdt.metadata.{FeatureInfo, InstanceInfo}
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTTree}
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.Maths

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import java.{util => ju}

import org.dma.gbdt4spark.algo.gbdt.helper.HistManager.NodeHist
import org.dma.gbdt4spark.algo.gbdt.histogram.{BinaryGradPair, GradPair, MultiGradPair}
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.loss.Loss
import org.dma.gbdt4spark.objective.metric.EvalMetric
import org.dma.gbdt4spark.objective.metric.EvalMetric.Kind
import org.dma.gbdt4spark.tree.split.SplitEntry

class DPGBDTTrainer(workerId: Int, val param: GBDTParam,
                    @transient private[gbdt] val featureInfo: FeatureInfo,
                    @transient private[gbdt] val trainData: Dataset[Int, Int],
                    @transient private[gbdt] val trainLabels: Array[Float],
                    @transient private[gbdt] val validData: Array[Vector],
                    @transient private[gbdt] val validLabels: Array[Float]) extends Serializable {
  private[gbdt] val forest = ArrayBuffer[GBTTree]()

  @transient private[gbdt] val validPreds = {
    if (param.numClass == 2)
      new Array[Float](validData.length)
    else
      new Array[Float](validData.length * param.numClass)
  }

  private[gbdt] val instanceInfo = InstanceInfo(param, trainData.numInstance)
  private[gbdt] val numFeatUsed = Math.round(featureInfo.numFeature * param.featSampleRatio)
  private[gbdt] val isFeatUsed = {
    if (numFeatUsed == featureInfo.numFeature)
      (0 until numFeatUsed).map(fid => featureInfo.getNumBin(fid) > 0).toArray
    else
      new Array[Boolean](numFeatUsed)
  }

//  private[gbdt] val activeNodes = ArrayBuffer[Int]()

  @transient private[gbdt] val histManager = HistManager(param, featureInfo)
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

  def createNewTree(): GradPair = {
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
    val sumGradPair = instanceInfo.calcGradPairs(trainLabels, loss, param, threadPool)
    tree.getRoot.setSumGradPair(sumGradPair)
    histManager.setGradPair(0, sumGradPair)
//    // 5. set root status
//    activeNodes += 0
    // 6. return root grad pair
    sumGradPair
  }

//  def buildHists(): Unit = {
//    val nids = activeNodes.clone()
//    val canSplits = nids.map(canSplitNode)
//    println(s"Nodes: [${(nids, canSplits).zipped.mkString(", ")}]")
//
//    val buildStart = System.currentTimeMillis()
//    var cur = 0
//    val toBuild = ArrayBuffer[Int]()
//    val toSubtract = ArrayBuffer[Boolean]()
//    while (cur < nids.length) {
//      val nid = nids(cur)
//      val sibNid = Maths.sibling(nid)
//      if (cur + 1 < nids.length && nids(cur + 1) == sibNid) {
//        if (canSplits(cur) || canSplits(cur + 1)) {
//          val curSize = instanceInfo.getNodeSize(nid)
//          val sibSize = instanceInfo.getNodeSize(sibNid)
//          if (curSize < sibSize) {
//            toBuild += nid
//            toSubtract += canSplits(cur + 1)
//          } else {
//            toBuild += sibNid
//            toSubtract += canSplits(cur)
//          }
//        } else {
//          histManager.removeNodeHist(Maths.parent(nid))
//        }
//        cur += 2
//      } else {
//        if (canSplits(cur)) {
//          toBuild += nid
//          toSubtract += false
//        }
//        cur += 1
//      }
//    }
//    println(s"To build ${toBuild.mkString(", ")}\nTo subtract ${toSubtract.mkString(", ")}")
//    timing {
//      if (toBuild.head == 0) {
//        histManager.buildHistForRoot(trainData, instanceInfo, threadPool)
//      } else {
//        histManager.buildHistForNodes(toBuild, trainData, instanceInfo, toSubtract, threadPool)
//      }
//    } {t => buildHistTime(nids.min) = t}
//    println(s"Build histograms cost ${System.currentTimeMillis() - buildStart} ms")
//  }

  def buildHists(toBuild: Seq[Int], toSubtract: Seq[Boolean]): Unit = {
    val buildStart = System.currentTimeMillis()
    timing {
      if (toBuild.head == 0) {
        histManager.buildHistForRoot(trainData, instanceInfo, threadPool)
      } else {
        histManager.buildHistForNodes(toBuild, trainData, instanceInfo, toSubtract, threadPool)
      }
    } {t => buildHistTime(toBuild.min) = t}
    println(s"Build histograms cost ${System.currentTimeMillis() - buildStart} ms")
  }

  def getNodeHists(nids: Seq[Int]): Seq[(Int, NodeHist)] = {
    nids.map(nid => (nid, histManager.getNodeHist(nid)))
  }

  def removeNodeHist(nid: Int): Unit = histManager.removeNodeHist(nid)

  def splitNodes(splits: Map[Int, SplitEntry]): Seq[(Int, Int)] = {
    val splitStart = System.currentTimeMillis()
//    val nids = activeNodes.clone()
//    activeNodes.clear()
//    nids.foreach(nid => {
//      if (splits.contains(nid))
//        splitNode(nid, splits(nid))
//      else
//        setAsLeaf(nid)
//    })
    val res = ArrayBuffer[(Int, Int)]()
    res.sizeHint(2 * splits.size)
    splits.foreach {
      case (nid, splitEntry) =>
        splitNode(nid, splitEntry)
        res += ((2 * nid + 1, instanceInfo.getNodeSize(2 * nid + 1)))
        res += ((2 * nid + 2, instanceInfo.getNodeSize(2 * nid + 2)))
    }
    println(s"Split nodes cost ${System.currentTimeMillis() - splitStart} ms")
    res
  }

  def splitNode(nid: Int, splitEntry: SplitEntry): Unit = {
    timing {
      instanceInfo.updatePos(nid, trainData, splitEntry,
        featureInfo.getSplits(splitEntry.getFid))
      val tree = forest.last
      val node = tree.getNode(nid)
      node.setSplitEntry(splitEntry)
      val leftChild = new GBTNode(2 * nid + 1, node, param.numClass)
      val rightChild = new GBTNode(2 * nid + 2, node, param.numClass)
      node.setLeftChild(leftChild)
      node.setRightChild(rightChild)
      tree.setNode(2 * nid + 1, leftChild)
      tree.setNode(2 * nid + 2, rightChild)
      val (leftGP, rightGP) = childrenGradPairs(nid, splitEntry)
      leftChild.setSumGradPair(leftGP)
      rightChild.setSumGradPair(rightGP)
      histManager.setGradPair(2 * nid + 1, leftGP)
      histManager.setGradPair(2 * nid + 2, rightGP)
      if (2 * nid + 1 >= Maths.pow(2, param.maxDepth) - 1) {
        histManager.removeNodeHist(nid)
        setAsLeaf(2 * nid + 1)
        setAsLeaf(2 * nid + 2)
      }
//      if (2 * nid + 1 < Maths.pow(2, param.maxDepth) - 1) {
//        activeNodes += 2 * nid + 1
//        activeNodes += 2 * nid + 2
//      } else {
//        histManager.removeNodeHist(nid)
//      }
    } (t => splitNodeTime(nid) = t)
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

  def childrenGradPairs(nid: Int, splitEntry: SplitEntry): (GradPair, GradPair) = {
    val histogram = histManager.getNodeHist(nid)(splitEntry.getFid)
    val sumGP = histManager.getGradPair(nid)
    val isCategorical = featureInfo.isCategorical(splitEntry.getFid)
    val splits = featureInfo.getSplits(splitEntry.getFid)
    val leftGP = sumGP.subtract(sumGP)
    for (i <- splits.indices) {
      if (splitEntry.flowTo(splits(i)) == 0)
        leftGP.plusBy(histogram.get(i))
    }
    if (isCategorical && splitEntry.defaultTo() == 0)
      leftGP.plusBy(histogram.get(splits.length))
    (leftGP, sumGP.subtract(leftGP))
  }

  def setAsLeaf(nid: Int): Unit = setAsLeaf(nid, forest.last.getNode(nid))

  def setAsLeaf(nid: Int, node: GBTNode): Unit = {
    node.chgToLeaf()
//    if (param.numClass == 2) {
//      val weight = node.calcWeight(param)
//      instanceInfo.updatePreds(nid, weight, param.learningRate)
//    } else {
//      val weights = node.calcWeights(param)
//      instanceInfo.updatePreds(nid, weights, param.learningRate)
//    }
    if (nid < Maths.pow(2, param.maxDepth) - 1)  // node not on the last level
      histManager.removeNodeHist(nid)
  }

//  def finishTree(): Unit = {
//    forest.last.getNodes.asScala.foreach {
//      case (nid, node) =>
//        if (node.getSplitEntry == null && !node.isLeaf)
//          setAsLeaf(nid, node)
//    }
//  }

  def finishTree(nodeGPs: Map[Int, GradPair]): Unit = {
    forest.last.getNodes.asScala.foreach {
      case (nid, node) =>
        require(node.getSplitEntry != null || node.isLeaf,
          s"Node[$nid] split[${node.getSplitEntry != null}] leaf[${node.isLeaf}]")
        if (node.isLeaf) {
          val gp = nodeGPs(nid)
          node.setSumGradPair(gp)
          if (param.numClass == 2) {
            val weight = node.calcWeight(param)
            instanceInfo.updatePreds(nid, weight, param.learningRate)
          } else {
            val weights = node.calcWeights(param)
            instanceInfo.updatePreds(nid, weights, param.learningRate)
          }
        }
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

    val evalMetrics = getEvalMetrics
    val metrics = evalMetrics.map(evalMetric => {
      val kind = evalMetric.getKind
      val trainSum = evalMetric.sum(instanceInfo.predictions, trainLabels)
      val trainMetric = kind match {
        case Kind.AUC => evalMetric.avg(trainSum, 1)
        case _ => evalMetric.avg(trainSum, trainLabels.length)
      }
      val validSum = evalMetric.sum(validPreds, validLabels)
      val validMetric = kind match {
        case Kind.AUC => evalMetric.avg(validSum, 1)
        case _ => evalMetric.avg(validSum, validLabels.length)
      }
      (kind, trainSum, trainMetric, validSum, validMetric)
    })
//    val metrics = evalMetrics.map(evalMetric => {
//      val kind = evalMetric.getKind
//      val trainSum = evalMetric.sum(instanceInfo.predictions, trainLabels)
//      val trainMetric = evalMetric.avg(trainSum, trainData.numInstance)
//      val validSum = evalMetric.sum(validPreds, validLabels)
//      val validMetric = evalMetric.avg(validSum, validLabels.length)
//        (kind, trainSum, trainMetric, validSum, validMetric)
//    })

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
