package org.dma.gbdt4spark.algo.gbdt.learner

import org.apache.spark.{SparkContext, SparkEnv}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.dma.gbdt4spark.algo.gbdt.histogram.Histogram
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{DataLoader, Maths, Transposer}

import scala.collection.mutable.ArrayBuffer

object SparkFPGBDTTrainer {
  var buildHistCost = 0L
  var histSubtractCost = 0L
  var findSplitCost = 0L
  var getSplitResultCost = 0L
  var retrieveCost = 0L
  var splitNodeCost = 0L

  def getCost = (SparkEnv.get.executorId,
    Seq("build" -> buildHistCost, "subtract" -> histSubtractCost,
      "find" -> findSplitCost, "result" -> getSplitResultCost,
      "retrieve" -> retrieveCost, "split" -> splitNodeCost))

  def clearCost = {
    buildHistCost = 0L
    histSubtractCost = 0L
    findSplitCost = 0L
    getSplitResultCost = 0L
    retrieveCost = 0L
    splitNodeCost = 0L
  }
}
import SparkFPGBDTTrainer._
class SparkFPGBDTTrainer(param: GBDTParam) extends Serializable {
  @transient implicit val sc = SparkContext.getOrCreate()

  type HistMap = collection.mutable.Map[Int, Array[Histogram]]
  @transient private var learners: RDD[(FPGBDTLearner, HistMap)] = _

  def loadData(input: String, validRatio: Double): Unit = {
    val loadStart = System.currentTimeMillis()
    val data = DataLoader.loadLibsvm(input, param.numFeature)
      .repartition(param.numWorker)
      .persist(StorageLevel.MEMORY_AND_DISK)
    val splits = data.randomSplit(Array(1.0 - validRatio, validRatio))
    val train = splits(0).cache()
    val valid = splits(1).cache()

    val numTrain = train.count()
    val numValid = valid.count()
    data.unpersist()
    println(s"load data cost ${System.currentTimeMillis() - loadStart} ms, " +
      s"$numTrain train data, $numValid valid data")

    val initStart = System.currentTimeMillis()
    val transposer = new Transposer()
    val (trainData, labels, bcFeatureInfo) = transposer.transpose2(train,
      param.numFeature, param.numWorker, param.numSplit)
    FPGBDTLearner.ensureLabel(labels, param.numClass)
    val bcLabels = sc.broadcast(labels)

    val bcParam = sc.broadcast(param)
    val learners = trainData.mapPartitionsWithIndex((partId, iterator) => Iterator((partId, iterator)))
      .zipPartitions(valid)(
        (iter, validIter) => {
          val (partId, trainIter) = iter.next()
          val valid = validIter.toArray
          val learner = new FPGBDTLearner(partId, bcParam.value,
            bcFeatureInfo.value, trainIter.toArray, bcLabels.value,
            valid.map(_.feature), valid.map(_.label.toFloat))
          val storedHists = collection.mutable.Map[Int, Array[Histogram]]()
          Iterator((learner, storedHists))
        }
      ).cache()
    learners.foreach(learner => println(s"Worker[${learner._1.workerId}] initialization done"))
    train.unpersist()
    valid.unpersist()
    this.learners = learners
    println(s"Transpose data and initialize workers cost ${System.currentTimeMillis() - initStart} ms")
  }

  def checkNodesOrder(nodes: Array[Int]): Unit = {
    if (nodes.length > 1) {
      val set = nodes.toSet
      for (i <- nodes.indices) {
        val nid = nodes(i)
        val siblingNid = Maths.sibling(nid)
        if (set.contains(siblingNid)) {
          if (nid < siblingNid)
            require(nodes(i + 1) == siblingNid, s"Failed: [${nodes.mkString(", ")}]")
          else
            require(nodes(i - 1) == siblingNid, s"Failed: [${nodes.mkString(", ")}]")
        }
      }
    }
  }

  def train(): Unit = {
    val trainStart = System.currentTimeMillis()

    val toFind = ArrayBuffer[Int]()
    val toSplit = collection.mutable.Map[Int, SplitEntry]()

    for (treeId <- 0 until param.numTree) {
      println(s"Start to train tree ${treeId + 1}")

      learners.foreach(_ => clearCost)

      toFind.clear()
      toSplit.clear()
      var curNodeNum = 1
      val maxNodeNum = param.maxNodeNum

      // create new tree
      val createStart = System.currentTimeMillis()
      learners.foreach(_._1.createNewTree())
      toFind += 0
      println(s"Tree[${treeId + 1}] Create new tree cost ${System.currentTimeMillis() - createStart} ms")

      // iteratively construct tree nodes
      while ((toFind.nonEmpty || toSplit.nonEmpty) && curNodeNum < maxNodeNum) {
        val startTime = System.currentTimeMillis()
        if (toFind.nonEmpty) {
          val nodes = toFind.toArray
          checkNodesOrder(nodes)
          val bcNodes = sc.broadcast(nodes)
          val res = learners.map {
            case (learner, storedHists) =>
              val cost = buildHistograms(bcNodes.value, learner, storedHists)
              val splits = findSplits(bcNodes.value, learner, storedHists)
              (cost, splits)
          }.collect()
          println(s"Tree[${treeId + 1}] nodes[${nodes.mkString(", ")}] " +
            s"executors decouple: [${res.map(_._1).mkString(", ")}] ms")
          val splits = res.map(_._2).reduce((splits1, splits2) => {
            (splits1, splits2).zipped.map((s1, s2) =>
              if (s1.needReplace(s2)) s2 else s1
            )
          })
//          val splits = learners.map {
//            case (learner, storedHists) =>
//              buildHistograms(bcNodes.value, learner, storedHists)
//              findSplits(bcNodes.value, learner, storedHists)
//          }.reduce((splits1, splits2) => {
//            (splits1, splits2).zipped.map((s1, s2) =>
//              if (s1.needReplace(s2)) s2 else s1
//            )
//          })
          (nodes, splits).zipped.foreach((nid, split) => toSplit += nid -> split)
          toFind.clear()
          println(s"Tree[${treeId + 1}] Build histograms " +
            s"and find splits for nodes[${nodes.mkString(", ")}] " +
            s"cost ${System.currentTimeMillis() - startTime} ms")
        } else {
          if (param.leafwise) {
            var bestNid = -1
            var bestSplit = null.asInstanceOf[SplitEntry]
            toSplit.toArray.foreach {
              case (nid, split) =>
                if (split.isEmpty || split.getGain <= param.minSplitGain) {
                  toSplit -= nid
                } else if (bestNid == -1 || bestSplit.needReplace(split)) {
                  bestNid = nid
                  bestSplit = split
                }
            }
            if (bestNid != -1) {
              val splitResult = learners.map {
                case (learner, _) => learner.getSplitResult(bestNid, bestSplit)
              }.filter(_.isDefined).collect()(0).get
              learners.foreach {
                case (learner, _) =>
                  learner.splitNode(bestNid, splitResult)
              }
              toSplit -= bestNid
              curNodeNum += 2
              if (2 * bestNid + 1 < Maths.pow(2, param.maxDepth) - 1) {
                toFind += 2 * bestNid + 1
                toFind += 2 * bestNid + 2
              }
              println(s"Tree[${treeId + 1}] Split node[$bestNid] " +
                s"cost ${System.currentTimeMillis() - startTime} ms")
            }
          } else {
            // TODO sort
            val validSplits = toSplit.filter {
              case (_, split) => !split.isEmpty && split.getGain > param.minSplitGain
            }.toArray
            val t0 = System.currentTimeMillis()
            if (validSplits.nonEmpty) {
              val bcValidSplits = sc.broadcast(validSplits)
              val splitResults = learners.flatMap {
                case (learner, _) =>
                  val getSplitStart = System.currentTimeMillis()
                  val res = bcValidSplits.value.map {
                    case (nid, split) =>
                      learner.getSplitResult(nid, split) match {
                        case Some(splitResult) => (nid, splitResult)
                        case None => null
                      }
                  }.filter(_ != null).iterator
                  getSplitResultCost += System.currentTimeMillis() - getSplitStart
                  res
              }.collect()
              val t1 = System.currentTimeMillis()
              val bcSplitResults = sc.broadcast(splitResults)
              learners.foreach {
                case (learner, _) =>
                  val retrieveStart = System.currentTimeMillis()
                  bcSplitResults.value.map(_._2.getRangeFrom)
                  val splitNodeStart = System.currentTimeMillis()
                  bcSplitResults.value.foreach {
                    case (nid, splitResult) => learner.splitNode(nid, splitResult)
                  }
                  retrieveCost += splitNodeStart - retrieveStart
                  splitNodeCost += System.currentTimeMillis() - splitNodeStart
              }
              toSplit.clear()
              curNodeNum += 2 * validSplits.length
              validSplits.foreach {
                case (nid, _) =>
                  if (2 * nid + 1 < Maths.pow(2, param.maxDepth) - 1) {
                    toFind += 2 * nid + 1
                    toFind += 2 * nid + 2
                  }
              }
              val t2 = System.currentTimeMillis()
              println(s"Tree[${treeId + 1}] Split nodes[${validSplits.map(_._1).mkString(", ")}] " +
                s"cost ${t1 - t0} + ${t2 - t1} = ${t2 - t0} ms")
//              println(s"Tree[${treeId + 1}] Split nodes[${validSplits.map(_._1).mkString(", ")}] " +
//                s"cost ${System.currentTimeMillis() - startTime} ms")
            }
          }
        }
      }

      // finish tree
      val finishStart = System.currentTimeMillis()
      val metrics = learners.map {
        case (learner, storedHists) =>
          storedHists.clear()
          learner.finishTree()
          learner.evaluate()
      }.collect()(0)
      val evalTrainMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
      println(s"Evaluation on train data after ${treeId + 1} tree(s): $evalTrainMsg")
      val evalValidMsg = metrics.map(metric => s"${metric._1}[${metric._3}]").mkString(", ")
      println(s"Evaluation on valid data after ${treeId + 1} tree(s): $evalValidMsg")
      println(s"Tree[${treeId + 1}] Finish tree cost ${System.currentTimeMillis() - finishStart} ms")

      println(s"${treeId + 1} tree(s) done, ${System.currentTimeMillis() - trainStart} ms elapsed")

      learners.map(_ => getCost)
        .collect()
        .sortBy(_._1)
        .foreach {
          case (partId, costs) =>
            println(s"Part[$partId] costs: $costs")
        }
    }

  }

  def buildHistograms(nodes: Array[Int], learner: FPGBDTLearner, storedHists: HistMap): Long = {
    val buildStart = System.currentTimeMillis()
    var cur = 0
    while (cur < nodes.length) {
      val nid = nodes(cur)
      val sibNid = Maths.sibling(nid)
      if (cur + 1 < nodes.length && nodes(cur + 1) == sibNid) {
        val curSize = learner.dataInfo.getNodeSize(nid)
        val sibSize = learner.dataInfo.getNodeSize(sibNid)
        val parHist = storedHists(Maths.parent(nid))
        val (curHist, sibHist) = if (curSize < sibSize) {
          val t0 = System.currentTimeMillis()
          val curHist = learner.buildHistograms(nid)
          val t1 = System.currentTimeMillis()
          learner.histSubtraction(parHist, curHist)
          val t2 = System.currentTimeMillis()
          buildHistCost += t1 - t0
          histSubtractCost += t2 - t1
          println(s"Node[$nid, $curSize] < Node[$sibNid, $sibSize], " +
            s"buildHist cost ${t1 - t0} ms, histSubtract cost ${t2 - t1} ms")
          (curHist, parHist)
        } else {
          val t0 = System.currentTimeMillis()
          val sibHist = learner.buildHistograms(sibNid)
          val t1 = System.currentTimeMillis()
          learner.histSubtraction(parHist, sibHist)
          val t2 = System.currentTimeMillis()
          buildHistCost += t1 - t0
          histSubtractCost += t2 - t1
          println(s"Node[$nid, $curSize] >= Node[$sibNid, $sibSize], " +
            s"buildHist cost ${t1 - t0} ms, histSubtract cost ${t2 - t1} ms")
          (parHist, sibHist)
        }
        storedHists += nid -> curHist
        storedHists += sibNid -> sibHist
        cur += 2
      } else {
        val t0 = System.currentTimeMillis()
        storedHists += nid -> learner.buildHistograms(nid)
        val t1 = System.currentTimeMillis()
        buildHistCost += t1 - t0
        println(s"Node[$nid, ${learner.dataInfo.getNodeSize(nid)}] " +
          s"buildHist cost ${t1 - t0} ms")
        cur += 1
      }
    }
    val buildCost = System.currentTimeMillis() - buildStart
    println(s"Build histograms for nodes[${nodes.mkString(", ")}] " +
      s"cost $buildCost ms")
    buildCost
  }

  def findSplits(nodes: Array[Int], learner: FPGBDTLearner, storedHists: HistMap): Array[SplitEntry] = {
    val t0 = System.currentTimeMillis()
    val res = nodes.map(nid => {
      learner.findLocalBestSplit(nid, storedHists(nid)).getSplitEntry
    })
    findSplitCost += System.currentTimeMillis() - t0
    res
  }

}
