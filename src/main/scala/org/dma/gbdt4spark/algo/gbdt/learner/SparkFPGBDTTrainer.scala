package org.dma.gbdt4spark.algo.gbdt.learner

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.dma.gbdt4spark.algo.gbdt.histogram.Histogram
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{DataLoader, Maths, Transposer}

import scala.collection.mutable.ArrayBuffer


class SparkFPGBDTTrainer(param: GBDTParam) extends Serializable {
  @transient implicit val sc = SparkContext.getOrCreate()

  type HistMap = collection.mutable.Map[Int, Array[Option[Histogram]]]
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
    val (trainDataFP, labels, bcFeatureInfo) = transposer.transpose(train,
      param.numFeature, param.numWorker, param.numSplit)
    FPGBDTLearner.ensureLabel(labels, param.numClass)
    val bcLabels = sc.broadcast(labels)

    val bcParam = sc.broadcast(param)

    val learners = trainDataFP.mapPartitionsWithIndex((partId, iterator) => Iterator((partId, iterator)))
      .zipPartitions(valid)(
        (iter, validIter) => {
          val (partId, trainIter) = iter.next()
          val valid = validIter.toArray
          val learner = new FPGBDTLearner(partId, bcParam.value,
            bcFeatureInfo.value, trainIter.toArray, bcLabels.value,
            valid.map(_.feature), valid.map(_.label.toFloat))
          val storedHists = collection.mutable.Map[Int, Array[Option[Histogram]]]()
          Iterator((learner, storedHists))
        }
      ).cache()
    learners.foreach(learner => println(s"Worker[${learner._1.workerId}] initialization done"))
    train.unpersist()
    valid.unpersist()
    this.learners = learners
    println(s"Transpose data and initialize workers cost ${System.currentTimeMillis() - initStart} ms")
  }

  def train(): Unit = {
    val trainStart = System.currentTimeMillis()

    val toFind = ArrayBuffer[Int]()
    val toSplit = collection.mutable.Map[Int, SplitEntry]()

    for (treeId <- 0 until param.numTree) {
      println(s"Start to train tree ${treeId + 1}")

      toFind.clear()
      toSplit.clear()
      var curNodeNum = 1
      val maxNodeNum = param.maxNodeNum

      // create new tree
      learners.foreach(_._1.createNewTree())
      toFind += 0

      // iteratively construct tree nodes
      while ((toFind.nonEmpty || toSplit.nonEmpty) && curNodeNum < maxNodeNum) {
        if (toFind.nonEmpty) {
          val nodes = toFind.toArray
          val bcNodes = sc.broadcast(nodes)
          val splits = learners.map {
            case (learner, storedHists) =>
              buildHistograms(bcNodes.value, learner, storedHists)
              findSplits(bcNodes.value, learner, storedHists)
          }.reduce((splits1, splits2) => {
            (splits1, splits2).zipped.map((s1, s2) =>
              if (s1.needReplace(s2)) s2
              else s1
            )
          })
          for (i <- nodes.indices)
            toSplit += nodes(i) -> splits(i)
          //(nodes, splits).zipped.foreach((nid, split) => toSplit += nid -> split)
          toFind.clear()
        } else {
          var bestNid = -1
          var bestSplit = null.asInstanceOf[SplitEntry]
          val leaves = ArrayBuffer[Int]()
          toSplit.toArray.foreach {
            case (nid, split) =>
              if (split.isEmpty || split.getGain <= param.minSplitGain) {
                leaves += nid
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
                leaves.foreach(learner.setAsLeaf)
            }
            toSplit -= bestNid
            if (2 * bestNid + 1 < Maths.pow(2, param.maxDepth) - 1) {
              toFind += 2 * bestNid + 1
              toFind += 2 * bestNid + 2
              curNodeNum += 2
            }
          } else {
            learners.foreach {
              case (learner, _) =>
                leaves.foreach(learner.setAsLeaf)
            }
          }
        }
      }

      // finish tree
      val metrics = learners.map {
        case (learner, storedHists) =>
          storedHists.clear()
          learner.finishTree()
          learner.evaluate()
      }.collect()(1)
      val evalTrainMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
      println(s"Evaluation on train data after ${treeId + 1} tree(s): $evalTrainMsg")
      val evalValidMsg = metrics.map(metric => s"${metric._1}[${metric._3}]").mkString(", ")
      println(s"Evaluation on valid data after ${treeId + 1} tree(s): $evalValidMsg")

      println(s"${treeId + 1} tree(s) done, ${System.currentTimeMillis() - trainStart} ms elapsed")
    }

  }

  def buildHistograms(nodes: Array[Int], learner: FPGBDTLearner, storedHists: HistMap): Unit = {
    var cur = 0
    while (cur < nodes.length) {
      val nid = nodes(cur)
      val sibNid = Maths.sibling(nid)
      if (cur + 1 < nodes.length && nodes(cur + 1) == sibNid) {
        val curSize = learner.dataInfo.getNodeSize(nid)
        val sibSize = learner.dataInfo.getNodeSize(sibNid)
        val parHist = storedHists(Maths.parent(nid))
        val (curHist, sibHist) = if (curSize < sibSize) {
          val curHist = learner.buildHistograms(nid)
          learner.histSubtraction(parHist, curHist)
          (curHist, parHist)
        } else {
          val sibHist = learner.buildHistograms(sibNid)
          learner.histSubtraction(parHist, sibHist)
          (parHist, sibHist)
        }
        storedHists += nid -> curHist
        storedHists += sibNid -> sibHist
        cur += 2
      } else {
        storedHists += nid -> learner.buildHistograms(nid)
        cur += 1
      }
    }
  }

  def findSplits(nodes: Array[Int], learner: FPGBDTLearner, storedHists: HistMap): Array[SplitEntry] = {
    nodes.map(nid => {
      learner.findLocalBestSplit(nid, storedHists(nid)).getSplitEntry
    })
  }

}
