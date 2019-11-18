package org.dma.gbdt4spark.algo.gbdt.trainer.csr

import org.apache.hadoop.fs.Path
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkConf, SparkContext, TaskContext}
import org.dma.gbdt4spark.algo.gbdt.dataset.Dataset
import org.dma.gbdt4spark.algo.gbdt.dataset.Dataset.{columnGrouping, createSketches, fromTextFile}
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTSplit, GBTTree}
import org.dma.gbdt4spark.common.Global.Conf._
import org.dma.gbdt4spark.data.Instance
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.metric.EvalMetric.Kind
import org.dma.gbdt4spark.sketch.HeapQuantileSketch
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.{DataLoader, Maths}

object SparkCSRGBDTTrainer {

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
      val trainer = new SparkCSRGBDTTrainer(param)
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

}

import org.dma.gbdt4spark.algo.gbdt.trainer.SparkFPGBDTTrainer._
class SparkCSRGBDTTrainer(param: GBDTParam) extends Serializable {

  @transient implicit val sc = SparkContext.getOrCreate()

  @transient private[gbdt] var bcFidToGroupId: Broadcast[Array[Int]] = _
  @transient private[gbdt] var bcGroupIdToFid: Broadcast[Array[Array[Int]]] = _
  @transient private[gbdt] var bcFidToNewFid: Broadcast[Array[Int]] = _
  @transient private[gbdt] var bcGroupSizes: Broadcast[Array[Int]] = _
  @transient private[gbdt] var bcFeatureInfo: Broadcast[FeatureInfo] = _

  @transient private[gbdt] var workers: RDD[CSRGBDTTrainerWrapper] = _

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
    val trainDP = fromTextFile(trainInput, numFeature)
      .coalesce(numWorker)
      .mapPartitions(iterator => Iterator(Dataset[Int, Float](iterator.toSeq)))
      .persist(StorageLevel.MEMORY_AND_DISK)
    val numTrain = trainDP.map(_.size).collect().sum
    println(s"Load data cost ${System.currentTimeMillis() - loadStart} ms")

    // 2. collect labels, ensure 0-based indexed and broadcast
    val labelStart = System.currentTimeMillis()
    val labels = new Array[Float](numTrain)
    val partLabels = trainDP.map(dataset =>
      (TaskContext.getPartitionId(), dataset.getLabels)
    ).collect()
    require(partLabels.map(_._1).distinct.length == partLabels.length
      && partLabels.map(_._2).forall(_.isDefined))
    var offset = 0
    partLabels.sortBy(_._1).map(_._2.get).foreach(partLabel => {
      Array.copy(partLabel, 0, labels, offset, partLabel.length)
      offset += partLabel.length
    })
    Instance.ensureLabel(labels, param.numClass)
    val bcLabels = sc.broadcast(labels)
    println(s"Collect labels cost ${System.currentTimeMillis() - labelStart} ms")

    // IdenticalPartitioner for shuffle operation
    class IdenticalPartitioner extends Partitioner {
      override def numPartitions: Int = numWorker

      override def getPartition(key: Any): Int = {
        val partId = key.asInstanceOf[Int]
        require(partId < numWorker, s"Partition id $partId exceeds maximum partition $numWorker")
        partId
      }
    }

    // 3. build quantile sketches, get candidate splits,
    // and create feature info, finally broadcast info to all workers
    val getSplitsStart = System.currentTimeMillis()
    val isCategorical = new Array[Boolean](numFeature)
    val splits = new Array[Array[Float]](numFeature)
    val featNNZ = new Array[Int](numFeature)
    trainDP.flatMap(dataset => {
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
    }).partitionBy(new IdenticalPartitioner)
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

    // 4. Partition features into groups,
    // get feature id to group id mapping and inverted indexing
    val featGroupStart = System.currentTimeMillis()
    val (fidToGroupId, groupIdToFid, groupSizes, fidToNewFid) = balancedFeatureGrouping(featNNZ, numWorker)
    val bcFidToGroupId = sc.broadcast(fidToGroupId)
    val bcGroupIdToFid = sc.broadcast(groupIdToFid)
    val bcFidToNewFid = sc.broadcast(fidToNewFid)
    val bcGroupSizes = sc.broadcast(groupSizes)
    println(s"Balanced feature grouping cost ${System.currentTimeMillis() - featGroupStart} ms")

    // 5. Perform horizontal-to-vertical partitioning
    val repartStart = System.currentTimeMillis()
    val trainFP = trainDP.flatMap(dataset => {
      // turn (feature index, feature value) into (feature index, bin index)
      // and partition into column groups
      columnGrouping(dataset, bcFidToGroupId.value,
        bcFidToNewFid.value, bcFeatureInfo.value, numWorker)
        .zipWithIndex.map {
        case (group, groupId) => (groupId, (TaskContext.getPartitionId(), group))
      }
    }).partitionBy(new IdenticalPartitioner)
      .mapPartitions(iterator => {
        // merge same group into a dataset
        val (groupIds, columnGroups) = iterator.toArray.unzip
        val groupId = groupIds.head
        require(groupIds.forall(_ == groupId))
        val partIds = columnGroups.map(_._1)
        require(partIds.distinct.length == partIds.length)
        Iterator(Dataset[Short, Byte](columnGroups.sortBy(_._1).map(_._2)))
      }).cache()
    require(trainFP.map(_.size).collect().forall(_ == numTrain))
    println(s"Repartitioning cost ${System.currentTimeMillis() - repartStart} ms")
    trainDP.unpersist() // persist FP and unpersist DP to save memory

    // 6. initialize worker
    val initWorkerStart = System.currentTimeMillis()
    val bcFeatNNZ = sc.broadcast(featNNZ)
    val valid = DataLoader.loadLibsvmDP(validInput, param.numFeature)
      .repartition(param.numWorker)
    val workers = trainFP.zipPartitions(valid, preservesPartitioning = true)(
      (trainIter, validIter) => {
        val workerId = TaskContext.getPartitionId
        val featureInfo = featureInfoOfGroup(bcFeatureInfo.value, workerId, bcGroupIdToFid.value(workerId))
        val groupFeatNNZ = new Array[Int](bcGroupIdToFid.value(workerId).length)
        for (fid <- 0 until bcParam.value.numFeature) {
          if (bcFidToGroupId.value(fid) == workerId) {
            val newFid = bcFidToNewFid.value(fid)
            groupFeatNNZ(newFid) = bcFeatNNZ.value(fid)
          }
        }

        val train = trainIter.toArray
        require(train.length == 1)
        val trainData = CSRDataset.apply(Dataset.restore(train.head), featureInfo, groupFeatNNZ)
        val trainLabels = bcLabels.value
        val valid = validIter.toArray
        val validData = valid.map(_.feature)
        val validLabels = valid.map(_.label.toFloat)
        Instance.ensureLabel(validLabels, bcParam.value.numClass)
        val worker = new CSRGBDTTrainer(workerId, bcParam.value, featureInfo,
          trainData, trainLabels, validData, validLabels)
        val wrapper = CSRGBDTTrainerWrapper(workerId, worker)
        Iterator(wrapper)
      }
    ).cache()
    workers.foreach(worker =>
      println(s"Worker[${worker.workerId}] initialization done"))
    val numValid = workers.map(_.validLabels.length).collect().sum
    trainFP.unpersist()
    println(s"Initialize workers done, cost ${System.currentTimeMillis() - initWorkerStart} ms, " +
      s"$numTrain train data, $numValid valid data")

    this.bcFidToGroupId = bcFidToGroupId
    this.bcGroupIdToFid = bcGroupIdToFid
    this.bcFidToNewFid = bcFidToNewFid
    this.bcGroupSizes = bcGroupSizes
    this.bcFeatureInfo = bcFeatureInfo
    this.workers = workers
    this.numTrain = numTrain
    this.numValid = numValid

    println(s"Initialization done, cost ${System.currentTimeMillis() - initStart} ms in total")

  }

  def train(): Seq[GBTTree] = {
    val trainStart = System.currentTimeMillis()

    val loss = ObjectiveFactory.getLoss(param.lossFunc)
    val evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(param.evalMetrics, loss)

    for (treeId <- 0 until param.numTree) {
      println(s"Start to train tree ${treeId + 1}")

      // 1. create new tree
      val createStart = System.currentTimeMillis()
      workers.foreach(_.createNewTree())
      val bestSplits = new Array[GBTSplit](Maths.pow(2, param.maxDepth) - 1)
      val bestOwnerIds = new Array[Int](Maths.pow(2, param.maxDepth) - 1)
      val bestAliasFids = new Array[Int](Maths.pow(2, param.maxDepth) - 1)
      println(s"Tree[${treeId + 1}] Create new tree cost ${System.currentTimeMillis() - createStart} ms")

      // 2. iteratively build one tree
      var hasActive = true
      while (hasActive) {
        // 2.1. build histograms and find local best splits
        val findStart = System.currentTimeMillis()
        val nids = collection.mutable.TreeSet[Int]()
        workers.map(worker => (worker.workerId, worker.findSplits()))
          .collect().foreach {
          case (workerId, splits) =>
            splits.foreach {
              case (nid, split) =>
                nids += nid
                if (bestSplits(nid) == null || bestSplits(nid).needReplace(split)) {
                  val fidInWorker = split.getSplitEntry.getFid
                  val trueFid = bcGroupIdToFid.value(workerId)(fidInWorker)
                  split.getSplitEntry.setFid(trueFid)
                  bestSplits(nid) = split
                  bestOwnerIds(nid) = workerId
                  bestAliasFids(nid) = fidInWorker
                }
            }
        }
        // (nid, ownerId, fidInWorker, split)
        val gatheredSplits = nids.toArray.map(nid => (nid,
          bestOwnerIds(nid), bestAliasFids(nid), bestSplits(nid)))
        val validSplits = gatheredSplits.filter(_._4.isValid(param.minSplitGain))
        val leaves = gatheredSplits.filter(!_._4.isValid(param.minSplitGain)).map(_._1)
        if (gatheredSplits.nonEmpty) {
          println(s"Build histograms and find best splits cost " +
            s"${System.currentTimeMillis() - findStart} ms, " +
            s"${validSplits.length} node(s) to split")
          val resultStart = System.currentTimeMillis()
          val bcValidSplits = sc.broadcast(validSplits)
          val bcLeaves = sc.broadcast(leaves)
          val splitResults = workers.flatMap(worker => {
            bcLeaves.value.foreach(worker.setAsLeaf)
            worker.getSplitResults(bcValidSplits.value).iterator
          }).collect()
          val bcSplitResults = sc.broadcast(splitResults)
          println(s"Get split results cost ${System.currentTimeMillis() - resultStart} ms")
          // 2.3. split nodes
          val splitStart = System.currentTimeMillis()
          hasActive = workers.map(_.splitNodes(bcSplitResults.value)).collect()(0)
          bcSplitResults.destroy()
          println(s"Split nodes cost ${System.currentTimeMillis() - splitStart} ms")
        } else {
          // no active nodes
          hasActive = false
        }
      }

      // 3. finish tree
      val finishStart = System.currentTimeMillis()
      val trainMetrics = new Array[Double](evalMetrics.length)
      val validMetrics = new Array[Double](evalMetrics.length)
      workers.map(worker => {
        worker.finishTree()
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
      println(s"Tree[${treeId + 1}] Finish tree cost ${System.currentTimeMillis() - finishStart} ms")

      val currentTime = System.currentTimeMillis()
      println(s"Train tree cost ${currentTime - createStart} ms, " +
        s"${treeId + 1} tree(s) done, ${currentTime - trainStart} ms elapsed")

      //      workers.map(_.reportTime()).collect().zipWithIndex.foreach {
      //        case (str, id) =>
      //          println(s"========Time cost summation of worker[$id]========")
      //          println(str)
      //      }
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
