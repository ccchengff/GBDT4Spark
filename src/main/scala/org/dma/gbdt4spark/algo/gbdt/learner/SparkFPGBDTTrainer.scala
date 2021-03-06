package org.dma.gbdt4spark.algo.gbdt.learner

import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo
import org.dma.gbdt4spark.algo.gbdt.tree.GBTSplit
import org.dma.gbdt4spark.data.{Instance, HorizontalPartition => HP, VerticalPartition => VP}
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.{DataLoader, Maths, Transposer}

class SparkFPGBDTTrainer(param: GBDTParam) extends Serializable {
  @transient implicit val sc = SparkContext.getOrCreate()

  @transient private var workers: RDD[FPGBDTLearner] = _

  def initialize(trainInput: String, validInput: String): Unit = {
    val bcParam = sc.broadcast(param)

    val loadStart = System.currentTimeMillis()
    val train = DataLoader.loadLibsvmFP(trainInput,
      param.numFeature, param.numWorker)
      .persist(StorageLevel.MEMORY_AND_DISK)
//    val train = DataLoader.loadLibsvmFP2(trainInput,
//      param.numFeature, param.numWorker)
//      .persist(StorageLevel.MEMORY_AND_DISK)
    val valid = DataLoader.loadLibsvmDP(validInput, param.numFeature)
      .repartition(param.numWorker)
      .persist(StorageLevel.MEMORY_AND_DISK)
    val numTrain = train.map(_.labels.length).reduce(_+_) / param.numWorker
    val numValid = valid.count()
    println(s"Load data cost ${System.currentTimeMillis() - loadStart} ms, " +
      s"$numTrain train data, $numValid valid data")

//    val getLabelStart = System.currentTimeMillis()
//    val labels = HP.getLabels(train)
//    val bcLabels = sc.broadcast(labels)
//    Instance.ensureLabel(labels, param.numClass)
//    println(s"Get labels cost ${System.currentTimeMillis() - getLabelStart} ms")

    val createFIStart = System.currentTimeMillis()
    val splits = new Array[Array[Float]](param.numFeature)
    train.mapPartitions(iterator =>
      VP.getCandidateSplits(iterator.toSeq,
        bcParam.value.numFeature, bcParam.value.numSplit).iterator
    ).collect().foreach {
      case (fid, fSplits) => splits(fid) = fSplits
    }
//    HP.getCandidateSplits(train, param.numFeature, param.numSplit, param.numWorker)
//      .foreach {
//        case (fid, fSplits) => splits(fid) = fSplits
//      }
    val featureInfo = FeatureInfo(param.numFeature, splits)
    val bcFeatureInfo = sc.broadcast(featureInfo)
    println(s"Create feature info cost ${System.currentTimeMillis() - createFIStart} ms")

    val initStart = System.currentTimeMillis()
//    val workers = HP.toVPDataset(train, labels.length, param.numFeature,
//      param.numWorker, bcFeatureInfo).zipPartitions(valid)(
//      (trainIter, validIter) => {
//        val train = trainIter.toArray
//        require(train.length == 1)
//        val trainData = train.head
//        val trainLabels = bcLabels.value
//        val valid = validIter.toArray
//        val validData = valid.map(_.feature)
//        val validLabels = valid.map(_.label.toFloat)
//        Instance.ensureLabel(validLabels, param.numClass)
//        val worker = new FPGBDTLearner(TaskContext.getPartitionId(),
//          bcParam.value, bcFeatureInfo.value,
//          trainData, trainLabels, validData, validLabels
//        )
//        Iterator(worker)
//      }
//    ).cache()
    val workers = train.zipPartitions(valid)(
      (vpIter, validIter) => {
        val (trainLabels, trainData) = VP.discretize(vpIter.toSeq, bcFeatureInfo.value)
        val valid = validIter.toArray
        val validLabels = valid.map(_.label.toFloat)
        val validData = valid.map(_.feature)
        Instance.ensureLabel(trainLabels, bcParam.value.numClass)
        Instance.ensureLabel(validLabels, bcParam.value.numClass)
        val worker = new FPGBDTLearner(TaskContext.getPartitionId,
          bcParam.value, bcFeatureInfo.value,
          trainData, trainLabels, validData, validLabels)
        Iterator(worker)
      }
    ).cache()
    workers.foreach(worker =>
      println(s"Worker[${worker.learnerId}] initialization done. " +
        s"Hyper-parameters:\n$param")
    )
    println(s"Initialize workers cost ${System.currentTimeMillis() - initStart} ms")

    train.unpersist()
    valid.unpersist()
    this.workers = workers
  }

  def loadData(input: String, validRatio: Double): Unit = {
    val loadStart = System.currentTimeMillis()
    val data = DataLoader.loadLibsvmDP(input, param.numFeature)
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
    Instance.ensureLabel(labels, param.numClass)
    val bcLabels = sc.broadcast(labels)

    val bcParam = sc.broadcast(param)
    val workers = trainData.zipPartitions(valid)(
        (trainIter, validIter) => {
          val learnerId = TaskContext.getPartitionId
          val valid = validIter.toArray
          val trainData = trainIter.toArray
          val trainLabels = bcLabels.value
          val validData = valid.map(_.feature)
          val validLabels = valid.map(_.label.toFloat)
          Instance.ensureLabel(validLabels, bcParam.value.numClass)
          val worker = new FPGBDTLearner(learnerId, bcParam.value, bcFeatureInfo.value,
            null, trainLabels, validData, validLabels)
          Iterator(worker)
        }
      ).cache()
    workers.foreach(worker =>
      println(s"Worker[${worker.learnerId}] initialization done. " +
        s"Hyper-parameters:\n$param")
    )

    train.unpersist()
    valid.unpersist()
    this.workers = workers
    println(s"Transpose data and initialize workers cost ${System.currentTimeMillis() - initStart} ms")
  }

  def train(): Unit = {
    val trainStart = System.currentTimeMillis()

    for (treeId <- 0 until param.numTree) {
      println(s"Start to train tree ${treeId + 1}")

      // 1. create new tree
      val createStart = System.currentTimeMillis()
      workers.foreach(_.createNewTree())
      val bestSplits = new Array[GBTSplit](Maths.pow(2, param.maxDepth) - 1)
      println(s"Tree[${treeId + 1}] Create new tree cost ${System.currentTimeMillis() - createStart} ms")

      var hasActive = true
      while (hasActive) {
        // 2. build histograms and find local best splits
        val findStart = System.currentTimeMillis()
        val nids = collection.mutable.TreeSet[Int]()
        workers.flatMap(_.findSplits().iterator)
          .collect()
          .foreach {
            case (nid, split) =>
              nids += nid
              if (bestSplits(nid) == null)
                bestSplits(nid) = split
              else
                bestSplits(nid).update(split)
          }
        val validSplits = nids.toArray.map(nid => (nid, bestSplits(nid)))
        println(s"Build histograms and find best splits cost " +
          s"${System.currentTimeMillis() - findStart} ms, " +
          s"${validSplits.length} node(s) to split")
        if (validSplits.nonEmpty) {
          // 3. get split results
          val resultStart = System.currentTimeMillis()
          val bcSplits = sc.broadcast(validSplits)
          val splitResults = workers.flatMap(
            _.getSplitResults(bcSplits.value).iterator
          ).collect()
          val bcSplitResults = sc.broadcast(splitResults)
          println(s"Get split results cost ${System.currentTimeMillis() - resultStart} ms")
          // 4. split nodes
          val splitStart = System.currentTimeMillis()
          hasActive = workers.map(_.splitNodes(bcSplitResults.value)).collect()(0)
          println(s"Split nodes cost ${System.currentTimeMillis() - splitStart} ms")
        } else {
          // no active nodes
          hasActive = false
        }
      }

      // 5. finish tree
      val finishStart = System.currentTimeMillis()
      val metrics = workers.map(worker => {
        worker.finishTree()
        worker.evaluate()
      }).collect()(0)
      val evalTrainMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
      println(s"Evaluation on train data after ${treeId + 1} tree(s): $evalTrainMsg")
      val evalValidMsg = metrics.map(metric => s"${metric._1}[${metric._3}]").mkString(", ")
      println(s"Evaluation on valid data after ${treeId + 1} tree(s): $evalValidMsg")
      println(s"Tree[${treeId + 1}] Finish tree cost ${System.currentTimeMillis() - finishStart} ms")

      val currentTime = System.currentTimeMillis()
      println(s"Train tree cost ${currentTime - createStart} ms, " +
        s"${treeId + 1} tree(s) done, ${currentTime - trainStart} ms elapsed")

      workers.map(_.reportTime()).collect().zipWithIndex.foreach {
        case (str, id) =>
          println(s"========Time cost summation of worker[$id]========")
          println(str)
      }
    }

    // TODO: check equality
    val forest = workers.map(_.finalizeModel()).collect()(0)
    forest.zipWithIndex.foreach {
      case (tree, treeId) =>
        println(s"Tree[${treeId + 1}] contains ${tree.size} nodes " +
          s"(${(tree.size - 1) / 2 + 1} leaves)")
    }
  }
}
