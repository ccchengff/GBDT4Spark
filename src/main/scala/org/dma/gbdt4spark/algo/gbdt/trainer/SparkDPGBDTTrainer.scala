package org.dma.gbdt4spark.algo.gbdt.trainer

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partitioner, SparkConf, SparkContext, TaskContext}
import org.apache.spark.storage.StorageLevel
import org.dma.gbdt4spark.algo.gbdt.dataset.Dataset._
import org.dma.gbdt4spark.algo.gbdt.helper.HistManager.NodeHist
import org.dma.gbdt4spark.algo.gbdt.helper.SplitFinder
import org.dma.gbdt4spark.algo.gbdt.histogram.{BinaryGradPair, GradPair, Histogram, MultiGradPair}
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTSplit, GBTTree}
import org.dma.gbdt4spark.common.Global.Conf._
import org.dma.gbdt4spark.data.Instance
import org.dma.gbdt4spark.exception.GBDTException
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.metric.EvalMetric.Kind
import org.dma.gbdt4spark.sketch.HeapQuantileSketch
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.{DataLoader, IdenticalPartitioner, Maths}

import scala.collection.mutable.{ArrayBuffer, Set => mutableSet}

object SparkDPGBDTTrainer {

  def main(args: Array[String]): Unit = {
    @transient val conf = new SparkConf()
    @transient implicit val sc = SparkContext.getOrCreate(conf)

    val param = new GBDTParam
    param.numClass = conf.getInt(ML_NUM_CLASS, DEFAULT_ML_NUM_CLASS)
    param.numFeature = conf.get(ML_NUM_FEATURE).toInt
    param.featSampleRatio = conf.getDouble(ML_FEATURE_SAMPLE_RATIO, DEFAULT_ML_FEATURE_SAMPLE_RATIO).toFloat
    param.numWorker = conf.get(ML_NUM_WORKER).toInt
    param.numThread = conf.getInt(ML_NUM_THREAD, DEFAULT_ML_NUM_THREAD)
    param.lossFunc = conf.get(ML_LOSS_FUNCTION)
    param.evalMetrics = conf.get(ML_EVAL_METRIC, DEFAULT_ML_EVAL_METRIC).split(",").map(_.trim).filter(_.nonEmpty)
    param.learningRate = conf.getDouble(ML_LEARN_RATE, DEFAULT_ML_LEARN_RATE).toFloat
    param.histSubtraction = conf.getBoolean(ML_GBDT_HIST_SUBTRACTION, DEFAULT_ML_GBDT_HIST_SUBTRACTION)
    param.lighterChildFirst = conf.getBoolean(ML_GBDT_LIGHTER_CHILD_FIRST, DEFAULT_ML_GBDT_LIGHTER_CHILD_FIRST)
    param.fullHessian = conf.getBoolean(ML_GBDT_FULL_HESSIAN, DEFAULT_ML_GBDT_FULL_HESSIAN)
    param.numSplit = conf.getInt(ML_GBDT_SPLIT_NUM, DEFAULT_ML_GBDT_SPLIT_NUM)
    param.numTree = conf.getInt(ML_GBDT_TREE_NUM, DEFAULT_ML_GBDT_TREE_NUM)
    param.maxDepth = conf.getInt(ML_GBDT_MAX_DEPTH, DEFAULT_ML_GBDT_MAX_DEPTH)
    val maxNodeNum = Maths.pow(2, param.maxDepth + 1) - 1
    param.maxNodeNum = conf.getInt(ML_GBDT_MAX_NODE_NUM, maxNodeNum) min maxNodeNum
    param.minChildWeight = conf.getDouble(ML_GBDT_MIN_CHILD_WEIGHT, DEFAULT_ML_GBDT_MIN_CHILD_WEIGHT).toFloat
    param.minNodeInstance = conf.getInt(ML_GBDT_MIN_NODE_INSTANCE, DEFAULT_ML_GBDT_MIN_NODE_INSTANCE)
    param.minSplitGain = conf.getDouble(ML_GBDT_MIN_SPLIT_GAIN, DEFAULT_ML_GBDT_MIN_SPLIT_GAIN).toFloat
    param.regAlpha = conf.getDouble(ML_GBDT_REG_ALPHA, DEFAULT_ML_GBDT_REG_ALPHA).toFloat
    param.regLambda = conf.getDouble(ML_GBDT_REG_LAMBDA, DEFAULT_ML_GBDT_REG_LAMBDA).toFloat max 1.0f
    param.maxLeafWeight = conf.getDouble(ML_GBDT_MAX_LEAF_WEIGHT, DEFAULT_ML_GBDT_MAX_LEAF_WEIGHT).toFloat
    println(s"Hyper-parameters:\n$param")

    val modelPath = conf.get(ML_MODEL_PATH)
    println(s"Model will be saved to $modelPath")

    try {
      val trainer = new SparkDPGBDTTrainer(param)
      val trainInput = conf.get(ML_TRAIN_DATA_PATH)
      val validInput = conf.get(ML_VALID_DATA_PATH)
      trainer.initialize(trainInput, validInput)
      val model = trainer.train()
      trainer.save(model, modelPath)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    } finally {
      //while (1 + 1 == 2) {}
    }
  }

  def balancedFeatureGrouping(numBins: Array[Int], numGroup: Int): (Array[Int], Array[Array[Int]]) = {
    val numFeature = numBins.length
    val fidToGroupId = new Array[Int](numFeature)
    val groupSizes = new Array[Int](numGroup)
    val groupNumBin = new Array[Long](numGroup)
    val sortedNumBins = numBins.zipWithIndex.sortBy(_._1)
    for (i <- 0 until (numFeature / 2)) {
      val fid = sortedNumBins(i)._2
      val groupId = fid % numGroup
      fidToGroupId(fid) = groupId
      groupSizes(groupId) += 1
      groupNumBin(groupId) += sortedNumBins(i)._1
    }
    for (i <- (numFeature / 2) until numFeature) {
      val fid = sortedNumBins(i)._2
      val groupId = numGroup - (fid % numGroup) - 1
      fidToGroupId(fid) = groupId
      groupSizes(groupId) += 1
      groupNumBin(groupId) += sortedNumBins(i)._1
    }
    val fidToNewFid = new Array[Int](numFeature)
    val groupIdToFid = groupSizes.map(groupSize => new Array[Int](groupSize))
    val curIndexes = new Array[Int](numGroup)
    for (fid <- fidToGroupId.indices) {
      val groupId = fidToGroupId(fid)
      val newFid = curIndexes(groupId)
      fidToNewFid(fid) = newFid
      groupIdToFid(groupId)(newFid) = fid
      curIndexes(groupId) += 1
    }
    println("Feature grouping info: " + (groupSizes, groupNumBin, 0 until numGroup).zipped.map {
      case (size, nnz, groupId) => s"(group[$groupId] #feature[$size] #nnz[$nnz])"
    }.mkString(" "))
    //(fidToGroupId, groupIdToFid, groupSizes, fidToNewFid)
    (fidToGroupId, groupIdToFid)
  }

  def buildScheme(nids: Seq[Int], nodeSizes: Array[Int], nodeGPs: Array[GradPair],
                  param: GBDTParam): (Seq[Int], Seq[Boolean], Seq[Int], Seq[Int]) = {
    val canSplits = nids.map(nid => {
      if (nodeSizes(nid) > param.minNodeInstance) {
        if (param.numClass == 2) {
          val sumGradPair = nodeGPs(nid).asInstanceOf[BinaryGradPair]
          param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
        } else {
          val sumGradPair = nodeGPs(nid).asInstanceOf[MultiGradPair]
          param.satisfyWeight(sumGradPair.getGrad, sumGradPair.getHess)
        }
      } else {
        false
      }
    })

    var cur = 0
    val toBuild = ArrayBuffer[Int]()
    val toSubtract = ArrayBuffer[Boolean]()
    val toRemove = ArrayBuffer[Int]()
    while (cur < nids.length) {
      val nid = nids(cur)
      val sibNid = Maths.sibling(nid)
      if (cur + 1 < nids.length && nids(cur + 1) == sibNid) {
        if (canSplits(cur) || canSplits(cur + 1)) {
          val curSize = nodeSizes(nid)
          val sibSize = nodeSizes(sibNid)
          if (curSize < sibSize) {
            toBuild += nid
            toSubtract += canSplits(cur + 1)
          } else {
            toBuild += sibNid
            toSubtract += canSplits(cur)
          }
        } else {
          toRemove += Maths.parent(nid)
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

    val toSetLeaf = (nids, canSplits).zipped.filter {
      case (_, canSplit) => !canSplit
    }._1

    (toBuild, toSubtract, toRemove, toSetLeaf)
  }

//  def histGrouping(histograms: Array[Histogram], fidToGroupId: Array[Int], groupIdToFid: Array[Array[Int]]): Array[(Int, Array[Histogram])] = {
//    //fidToGroupId, groupIdToFid
//    val res = groupIdToFid.map(fids => new Array[Histogram](fids.length))
//    val indexes = new Array[Int](groupIdToFid.length)
//    for (fid <- histograms.indices) {
//      val groupId = fidToGroupId(fid)
//      res(groupId)(indexes(groupId)) = histograms(fid)
//      indexes(groupId) += 1
//    }
//    res.zipWithIndex.map {
//      case (hists, groupId) => (groupId, hists)
//    }
//  }
//
//  def findSplit(partHists: Array[Array[Histogram]], featureInfo: FeatureInfo,
//                groupId: Int, fids: Array[Int], nodeGP: GradPair, param: GBDTParam): GBTSplit = {
//    // aggregate hists
//    val first = partHists.head
//    for (i <- 1 until partHists.length) {
//      val plus = partHists(i)
//      for (j <- first.indices) {
//        if (first(j) != null) {
//          first(j).plusBy(plus(j))
//        }
//      }
//    }
//    val hists = new Array[Histogram](featureInfo.numFeature)
//    for (i <- fids.indices) {
//      hists(fids(i)) = first(i)
//    }
//
//    // find split
//    val splitFinder = new SplitFinder(param, featureInfo)
//    val nodeGain = nodeGP.calcGain(param)
//    splitFinder.findBestSplit(hists, nodeGP, nodeGain)
//  }

  def findSplit(nodeHists: Array[NodeHist], featureInfo: FeatureInfo,
                nodeGradPair: GradPair, param: GBDTParam): GBTSplit = {
    // aggregate hists
    val hist = nodeHists.head
    for (i <- 1 until nodeHists.length) {
      val plus = nodeHists(i)
      for (j <- hist.indices) {
        if (hist(j) != null) {
          hist(j).plusBy(plus(j))
        }
      }
    }

    // find split
    val splitFinder = new SplitFinder(param, featureInfo)
    val nodeGain = nodeGradPair.calcGain(param)
    splitFinder.findBestSplit(hist, nodeGradPair, nodeGain)
  }

}

import SparkDPGBDTTrainer._
class SparkDPGBDTTrainer(param: GBDTParam) extends Serializable {
  @transient implicit val sc = SparkContext.getOrCreate()

  private[gbdt] var bcParam: Broadcast[GBDTParam] = _
  private[gbdt] var bcFeatureInfo: Broadcast[FeatureInfo] = _
  private[gbdt] var workers: RDD[DPGBDTTrainerWrapper] = _

  //private[gbdt] var bcFidToGroupId: Broadcast[Array[Int]] = _
  //private[gbdt] var bcGroupIdToFid: Broadcast[Array[Array[Int]]] = _

  private[gbdt] var numTrain: Int = _
  private[gbdt] var numValid: Int = _

  def initialize(trainInput: String, validInput: String)
                (implicit sc: SparkContext): Unit = {
    val initStart = System.currentTimeMillis()
    val bcParam = sc.broadcast(param)
    val numFeature = param.numFeature
    val numWorker = param.numWorker
    val numSplit = param.numSplit

    // 1. load data from hdfs
    val loadStart = System.currentTimeMillis()
    val train = DataLoader.loadLibsvmDP(trainInput, numFeature)
      .repartition(numWorker)
      .mapPartitions(iterator => Iterator(fromLabeledData(iterator.toArray)))
      .persist(StorageLevel.MEMORY_AND_DISK)
//    val train = fromTextFile(trainInput, numFeature)
//      .repartition(numWorker)
//      .mapPartitions(iterator => Iterator(Dataset[Int, Float](iterator.toSeq)))
//      .persist(StorageLevel.MEMORY_AND_DISK)
    val numTrain = train.map(_.size).collect().sum
    println(s"Load data cost ${System.currentTimeMillis() - loadStart} ms")

    train.foreach(dataset => println(s"Worker[${TaskContext.getPartitionId()}] " +
      s"have ${dataset.numPartition} partitions, ${dataset.numInstance} instances, " +
      s"is labeled [${dataset.getLabels.isDefined}]"))

//    // IdenticalPartitioner for shuffle operation
//    class IdenticalPartitioner extends Partitioner {
//      override def numPartitions: Int = numWorker
//
//      override def getPartition(key: Any): Int = {
//        val partId = key.asInstanceOf[Int]
//        require(partId < numWorker, s"Partition id $partId exceeds maximum partition $numWorker")
//        partId
//      }
//    }

    // 2. build quantile sketches, get candidate splits,
    // and create feature info, finally broadcast info to all workers
    val getSplitsStart = System.currentTimeMillis()
    val isCategorical = new Array[Boolean](numFeature)
    val splits = new Array[Array[Float]](numFeature)
    val featNNZ = new Array[Int](numFeature)
    train.flatMap(dataset => {
      val sketchGroups = new Array[Array[HeapQuantileSketch]](numWorker)
      (0 until numWorker).foreach(groupId => {
        val groupSize = numFeature / numWorker + (if (groupId < (numFeature % numWorker)) 1 else 0)
        sketchGroups(groupId) = new Array[HeapQuantileSketch](groupSize)
      })
      val sketches = createSketches(dataset, numFeature)
      val curIndex = new Array[Int](numWorker)
      for (fid <- 0 until numFeature) {
        val groupId = fid % numWorker
        if (sketches(fid) == null || sketches(fid).isEmpty) {
          sketchGroups(groupId)(curIndex(groupId)) = null
        } else {
          sketchGroups(groupId)(curIndex(groupId)) = sketches(fid)
        }
        curIndex(groupId) += 1
      }
      sketchGroups.zipWithIndex.map {
        case (group, groupId) => (groupId, group)
      }.iterator
    }).partitionBy(new IdenticalPartitioner(numWorker))
      .mapPartitions(iterator => {
        // merge quantile sketches and get quantiles as candidate splits
        val (groupIds, sketchGroups) = iterator.toArray.unzip
        val groupId = groupIds.head
        require(groupIds.forall(_ == groupId))
        val merged = sketchGroups.head
        val tail = sketchGroups.tail
        val size = merged.length
        val splits = (0 until size).map(i => {
          tail.foreach(group => {
            if (merged(i) == null || merged(i).isEmpty) {
              merged(i) = group(i)
            } else {
              merged(i).merge(group(i))
            }
          })
          if (merged(i) != null && !merged(i).isEmpty) {
            val distinct = merged(i).tryDistinct(FeatureInfo.ENUM_THRESHOLD)
            if (distinct == null)
              (false, Maths.unique(merged(i).getQuantiles(numSplit)), merged(i).getN.toInt)
            else
              (true, distinct, merged(i).getN.toInt)
          } else {
            (false, null, 0)
          }
        })
        Iterator((groupId, splits))
      }, preservesPartitioning = true)
      .collect()
      .foreach {
        case (groupId, groupSplits) =>
          // restore feature id based on column grouping info
          // and set splits to corresponding feature
          groupSplits.view.zipWithIndex.foreach {
            case ((fIsCategorical, fSplits, nnz), index) =>
              val fid = index * numWorker + groupId
              isCategorical(fid) = fIsCategorical
              splits(fid) = fSplits
              featNNZ(fid) = nnz
          }
      }
    val featureInfo = FeatureInfo(isCategorical, splits)
    val bcFeatureInfo = sc.broadcast(featureInfo)
    println(s"Create feature info cost ${System.currentTimeMillis() - getSplitsStart} ms")

    // 3.
    //val (fidToGroupId, groupIdToFid) = balancedFeatureGrouping(featureInfo.numBin, numWorker)

    // 4. initialize worker
    val initWorkerStart = System.currentTimeMillis()
    val valid = DataLoader.loadLibsvmDP(validInput, numFeature)
      .repartition(numWorker)
    val workers = train.zipPartitions(valid, preservesPartitioning = true)(
      (trainIter, validIter) => {
        val train = trainIter.toArray
        require(train.length == 1)
        val trainData = binning(train.head, bcFeatureInfo.value)
        val trainLabels = train.head.getLabels.get
        Instance.ensureLabel(trainLabels, bcParam.value.numClass)
        val valid = validIter.toArray
        val validData = valid.map(_.feature)
        val validLabels = valid.map(_.label.toFloat)
        Instance.ensureLabel(validLabels, bcParam.value.numClass)
        val workerId = TaskContext.getPartitionId()
        val worker = new DPGBDTTrainer(workerId, bcParam.value, bcFeatureInfo.value,
          trainData, trainLabels, validData, validLabels)
        val wrapper = DPGBDTTrainerWrapper(workerId, worker)
        Iterator(wrapper)
      }
    ).cache()
    workers.foreach(worker =>
      println(s"Worker[${worker.workerId}] initialization done"))
    val numValid = workers.map(_.validLabels.length).collect().sum
    train.unpersist()
    println(s"Initialize workers done, cost ${System.currentTimeMillis() - initWorkerStart} ms, " +
      s"$numTrain train data, $numValid valid data")
    println(s"Number of train data on each worker: " +
      s"[${workers.map(_.trainLabels.length).collect().mkString(", ")}]")

    this.bcParam = bcParam
    this.bcFeatureInfo = bcFeatureInfo
    this.workers = workers
    this.numTrain = numTrain
    this.numValid = numValid

    //this.bcFidToGroupId = sc.broadcast(fidToGroupId)
    //this.bcGroupIdToFid = sc.broadcast(groupIdToFid)

    println(s"Initialization done, cost ${System.currentTimeMillis() - initStart} ms in total")
  }

  def train(): Seq[GBTTree] = {
    val trainStart = System.currentTimeMillis()

    val loss = ObjectiveFactory.getLoss(param.lossFunc)
    val evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(param.evalMetrics, loss)

    val partitioner = new Partitioner {
      override def numPartitions: Int = bcParam.value.numWorker

      override def getPartition(key: Any): Int = key.asInstanceOf[Int] % numPartitions
    }
    //val partitioner = new IdenticalPartitioner(param.numWorker)

    val maxActiveNid = Maths.pow(2, param.maxDepth) - 1

    for (treeId <- 0 until param.numTree) {
      println(s"Start to train tree ${treeId + 1}")

      // 1. create new tree
      val createStart = System.currentTimeMillis()
      val nodeSizes = new Array[Int](Maths.pow(2, param.maxDepth + 1) - 1)
      val nodeGPs = new Array[GradPair](Maths.pow(2, param.maxDepth + 1) - 1)
      val treeLeaves = ArrayBuffer[Int]()
      val activeNodes = ArrayBuffer[Int]()
      nodeSizes(0) = numTrain
      nodeGPs(0) = workers.map(_.createNewTree()).reduce((gp1, gp2) => gp1.plus(gp2))
      activeNodes += 0
      println(s"Tree[${treeId + 1}] Create new tree cost ${System.currentTimeMillis() - createStart} ms")

      // 2. iteratively build one tree
      var hasActive = true
      while (hasActive) {
        val buildStart = System.currentTimeMillis()
        val scheme = buildScheme(activeNodes, nodeSizes, nodeGPs, param)
        treeLeaves ++= scheme._4
        val bcScheme = sc.broadcast(scheme)
        workers.foreach(worker => {
          val (toBuild, toSubtract, toRemove, toSetLeaf) = bcScheme.value
          toRemove.foreach(worker.removeNodeHist)
          worker.buildHists(toBuild, toSubtract)
          toSetLeaf.foreach(worker.setAsLeaf)
        })
        println(s"Build histograms cost ${System.currentTimeMillis() - buildStart} ms")

        val findStart = System.currentTimeMillis()
        val toSplit = activeNodes.toSet.diff(scheme._4.toSet)
        val bcNodeGPs = sc.broadcast(
          toSplit.map(nid => (nid, nodeGPs(nid))).toMap
        )
//        val splits = workers.flatMap(worker =>
//          worker.getNodeHists(bcNodeGPs.value.keys.toSeq)
//            .mapValues(nodeHists => NodeHistWrapper(nodeHists, bcParam.value.numClass, bcParam.value.fullHessian))
//        )
        val splits = workers.mapPartitions(iterator => {
          val workers = iterator.toArray
          require(workers.length == 1)
          workers.head.getNodeHists(bcNodeGPs.value.keys.toSeq)
              .map {
                case (nid, nodeHists) => (nid, NodeHistWrapper(nodeHists,
                  bcParam.value.numClass, bcParam.value.fullHessian))
              }.iterator
        }).partitionBy(partitioner)
          .mapPartitions(iterator => {
            iterator.toArray.groupBy(_._1).map {
              case (nid, nodeHists) =>
                require(nodeHists.forall(_._1 == nid))
                val split = findSplit(nodeHists.map(_._2.histograms),
                  bcFeatureInfo.value, bcNodeGPs.value(nid), bcParam.value)
                (nid, split)
            }.iterator
          }, preservesPartitioning = true)
          .collect()
        val validSplits = splits.filter(_._2.isValid(param.minSplitGain))
        validSplits.foreach {
          case (nid, split) =>
            nodeGPs(2 * nid + 1) = split.getLeftGradPair
            nodeGPs(2 * nid + 2) = split.getRightGradPair
        }
        val leaves = splits.filter(!_._2.isValid(param.minSplitGain)).map(_._1)
        treeLeaves ++= leaves
        println(s"Find splits cost ${System.currentTimeMillis() - findStart} ms")

        activeNodes.clear()
        if (splits.nonEmpty) {
          val splitStart = System.currentTimeMillis()
          val bcValidSplits = sc.broadcast(validSplits.map(s => (s._1, s._2.getSplitEntry)))
          val bcLeaves = sc.broadcast(leaves)
          val toSetActive = mutableSet[Int]()
          val toSetLeaf = mutableSet[Int]()
          workers.flatMap(worker => {
            bcLeaves.value.foreach(worker.setAsLeaf)
            worker.splitNodes(bcValidSplits.value.toMap).iterator
          }).collect().foreach {
            case (nid, size) =>
              nodeSizes(nid) += size
              if (nid < maxActiveNid)
                toSetActive += nid
              else
                toSetLeaf += nid
          }
          activeNodes ++= toSetActive.toSeq.sorted
          treeLeaves ++= toSetLeaf.toSeq
          println(s"Split nodes cost ${System.currentTimeMillis() - splitStart} ms")
        }
        hasActive = activeNodes.nonEmpty
      }

      // 3. finish tree
      val finishStart = System.currentTimeMillis()
      val bcGPs = sc.broadcast(treeLeaves.map(nid => (nid, nodeGPs(nid))).toMap)
      val trainMetrics = new Array[Double](evalMetrics.length)
      val validMetrics = new Array[Double](evalMetrics.length)
      workers.map(worker => {
        worker.finishTree(bcGPs.value)
        worker.evaluate()
      }).collect().foreach(_.zipWithIndex.foreach {
        case ((kind, train, valid), index) =>
          require(kind == evalMetrics(index).getKind)
          trainMetrics(index) += train
          validMetrics(index) += valid
      })
      val evalTrainMsg = (evalMetrics, trainMetrics).zipped.map {
        case (evalMetric, trainSum) => evalMetric.getKind match {
          case Kind.AUC => s"${evalMetric.getKind}[${evalMetric.avg(trainSum, workers.count.toInt)}]"
          case _ => s"${evalMetric.getKind}[${evalMetric.avg(trainSum, numTrain)}]"
        }
      }.mkString(", ")
      println(s"Evaluation on train data after ${treeId + 1} tree(s): $evalTrainMsg")
      val evalValidMsg = (evalMetrics, validMetrics).zipped.map {
        case (evalMetric, validSum) => evalMetric.getKind match {
          case Kind.AUC => s"${evalMetric.getKind}[${evalMetric.avg(validSum, workers.count.toInt)}]"
          case _ => s"${evalMetric.getKind}[${evalMetric.avg(validSum, numValid)}]"
        }
      }.mkString(", ")
      println(s"Evaluation on valid data after ${treeId + 1} tree(s): $evalValidMsg")
//      val evalTrainMsg = (evalMetrics, trainMetrics).zipped.map {
//        case (evalMetric, trainSum) => s"${evalMetric.getKind}[${evalMetric.avg(trainSum, numTrain)}]"
//      }.mkString(", ")
//      println(s"Evaluation on train data after ${treeId + 1} tree(s): $evalTrainMsg")
//      val evalValidMsg = (evalMetrics, validMetrics).zipped.map {
//        case (evalMetric, validSum) => s"${evalMetric.getKind}[${evalMetric.avg(validSum, numValid)}]"
//      }.mkString(", ")
//      println(s"Evaluation on valid data after ${treeId + 1} tree(s): $evalValidMsg")
      println(s"Tree[${treeId + 1}] Finish tree cost ${System.currentTimeMillis() - finishStart} ms")

      val currentTime = System.currentTimeMillis()
      println(s"Train tree cost ${currentTime - createStart} ms, " +
        s"${treeId + 1} tree(s) done, ${currentTime - trainStart} ms elapsed")
    }

    // TODO: check equality
    val forest = workers.map(_.finalizeModel()).collect()(0)
    forest.zipWithIndex.foreach {
      case (tree, treeId) =>
        println(s"Tree[${treeId + 1}] contains ${tree.size} nodes " +
          s"(${(tree.size - 1) / 2 + 1} leaves)")
    }
    forest
  }

  def save(model: Seq[GBTTree], modelPath: String)(implicit sc: SparkContext): Unit = {
    val path = new Path(modelPath)
    val fs = path.getFileSystem(sc.hadoopConfiguration)
    if (fs.exists(path)) fs.delete(path, true)
    sc.parallelize(Seq(model)).saveAsObjectFile(modelPath)
  }

}

case class NodeHistWrapper(var histograms: Array[Histogram], var numClass: Int, var fullHessian: Boolean) {

  def writeObject(oos: ObjectOutputStream): Unit = {
    oos.writeInt(numClass)
    oos.writeInt(if (fullHessian) 1 else 0)
    oos.writeInt(histograms.length)
    if (numClass == 2) {
      for (hist <- histograms) {
        if (hist == null) {
          oos.writeInt(0)
        } else {
          val numBin = hist.getNumBin
          oos.writeInt(numBin)
          val gradients = hist.getGradients
          val hessians = hist.getHessians
          for (i <- 0 until numBin) {
            oos.writeDouble(gradients(i))
            oos.writeDouble(hessians(i))
          }
        }
      }
    } else if (!fullHessian) {
      for (hist <- histograms) {
        if (hist == null) {
          oos.writeInt(0)
        } else {
          val numBin = hist.getNumBin
          oos.writeInt(numBin)
          val gradients = hist.getGradients
          val hessians = hist.getHessians
          for (i <- 0 until numBin) {
            val offset = i * numClass
            for (k <- 0 until numClass) {
              oos.writeDouble(gradients(offset + k))
              oos.writeDouble(hessians(offset + k))
            }
          }
        }
      }
    } else {
      throw new GBDTException("Full hessian not supported")
    }
  }

  def readObject(ois: ObjectInputStream): Unit = {
    this.numClass = ois.readInt()
    this.fullHessian = ois.readInt() == 1
    val numFeature = ois.readInt()
    this.histograms = new Array[Histogram](numFeature)
    if (numClass == 2) {
      for (fid <- 0 until numFeature) {
        val numBin = ois.readInt()
        if (numBin > 0) {
          val gradients = new Array[Double](numBin)
          val hessians = new Array[Double](numBin)
          for (i <- 0 until numBin) {
            gradients(i) = ois.readDouble()
            hessians(i) = ois.readDouble()
          }
          histograms(fid) = new Histogram(numBin, numClass, fullHessian, gradients, hessians)
        }
      }
    } else if (!fullHessian) {
      for (fid <- 0 until numFeature) {
        val numBin = ois.readInt()
        if (numBin > 0) {
          val gradients = new Array[Double](numBin * numClass)
          val hessians = new Array[Double](numBin * numClass)
          for (i <- 0 until numBin) {
            val offset = i * numClass
            for (k <- 0 until numClass) {
              gradients(offset + k) = ois.readDouble()
              hessians(offset + k) = ois.readDouble()
            }
          }
          histograms(fid) = new Histogram(numBin, numClass, fullHessian, gradients, hessians)
        }
      }
    } else {
      throw new GBDTException("Full hessian not supported")
    }
  }
}