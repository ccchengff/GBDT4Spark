package org.dma.gbdt4spark.algo.gbdt

import it.unimi.dsi.fastutil.bytes.ByteArrayList
import it.unimi.dsi.fastutil.ints.IntArrayList
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.dma.gbdt4spark.algo.gbdt.GBDTPhase.GBDTPhase
import org.dma.gbdt4spark.algo.gbdt.histogram._
import org.dma.gbdt4spark.algo.gbdt.metadata.{DataInfo, FeatureInfo}
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTSplit, GBTTree}
import org.dma.gbdt4spark.data.{FeatureRow, Instance}
import org.dma.gbdt4spark.exception.GBDTException
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.objective.loss.Loss
import org.dma.gbdt4spark.objective.metric.EvalMetric
import org.dma.gbdt4spark.sketch.HeapQuantileSketch
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.{DataLoader, EvenPartitioner, Maths}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Sorting

object GBDTTrainer {
  private val LOG = LoggerFactory.getLogger(GBDTTrainer.getClass)

  def apply(param: GBDTParam): GBDTTrainer = new GBDTTrainer(param)

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
      LOG.warn(s"Change range of labels from [1, $numLabel] to [0, ${numLabel - 1}]")
      for (i <- labels.indices)
        labels(i) -= 1
    }
  }

  private var globalClock: Int = 0
  private var workerClock: Array[Int] = _
  private var numLocalParts: Int = 0

  private var dataInfo: DataInfo = _
  private var loss: Loss = _
  private var evalMetrics: Array[EvalMetric] = _

  type NodeHistMap = collection.mutable.Map[Int, Array[Option[Histogram]]]
  private var nodeHists: collection.mutable.Map[Int, NodeHistMap] = _

  def sync[A](localId: Int)(f: => A): A = {
    var res: A = null.asInstanceOf[A]
    if (globalClock == workerClock(localId)) {
      GBDTTrainer.synchronized {
        if (globalClock == workerClock(localId)) {
          res = f
          globalClock += 1
        }
      }
    } else if (globalClock < workerClock(localId)) {
      throw new GBDTException(s"Invalid clock: global[$globalClock] worker[${workerClock(localId)}]")
    }
    workerClock(localId) += 1
    res
  }
}


class GBDTTrainer(@transient val param: GBDTParam) extends Serializable {
  private val LOG = GBDTTrainer.LOG

  // environment parameter
  @transient private val spark = SparkSession.builder().getOrCreate()
  private val bcParam = spark.sparkContext.broadcast(param)

  // train data and valid data
  @transient private var trainData: RDD[Option[FeatureRow]] = _
  @transient private var validData: RDD[Instance] = _
  @transient private var bcNumTrainData: Broadcast[Int] = _
  @transient private var bcNumValidData: Broadcast[Int] = _

  // broadcast variables
  @transient private var bcFeatureEdges: Broadcast[Array[Int]] = _
  @transient private var bcFeatureInfo: Broadcast[FeatureInfo] = _
  @transient private var bcLabels: Broadcast[Array[Float]] = _

  // RDD to control partitions, all partitions have the same data info
  @transient private var partitions: RDD[(Int, Int, Array[Int])] = _

  // tree info
  @transient private var forest: ArrayBuffer[GBTTree] = _
  @transient private var phase: GBDTPhase = _
  @transient private var toBuild: collection.mutable.Map[Int, Int] = _
  @transient private var toFind: collection.mutable.Set[Int] = _
  @transient private var toSplit: collection.mutable.Map[Int, GBTSplit] = _
  @transient private var storedNodeHists: collection.mutable.Set[Int] = _

  def loadData(input: String, validRatio: Double): Unit = {
    val loadStart = System.currentTimeMillis()
    // 1. load original data, split into train data and valid data
    val dim = param.numFeature
    val fullDataset = spark.sparkContext.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => DataLoader.parseLibsvm(line, dim))
      .persist(StorageLevel.MEMORY_AND_DISK)
    val dataset = fullDataset.randomSplit(Array[Double](1.0 - validRatio, validRatio))
    val oriTrainData = dataset(0).persist(StorageLevel.MEMORY_AND_DISK)
    val validData = dataset(1).persist(StorageLevel.MEMORY_AND_DISK)
    val bcNumTrainData = spark.sparkContext.broadcast(oriTrainData.count().toInt)
    val bcNumValidData = spark.sparkContext.broadcast(validData.count().toInt)
    fullDataset.unpersist()
    LOG.info(s"Load data cost ${System.currentTimeMillis() - loadStart} ms, " +
      s"${bcNumTrainData.value} train data, ${bcNumValidData.value} valid data")
    // start to transpose train data
    val transposeStart = System.currentTimeMillis()
    // 2. get #instance of each partition, and calculate the offsets of instance indexes
    val oriNumPart = oriTrainData.getNumPartitions
    val partNumInstance = new Array[Int](oriNumPart)
    oriTrainData.mapPartitionsWithIndex((partId, iterator) =>
      Seq((partId, iterator.size)).iterator)
      .collect()
      .foreach(part => partNumInstance(part._1) = part._2)
    val partInsIdOffset = new Array[Int](oriNumPart)
    for (i <- 1 until oriNumPart)
      partInsIdOffset(i) += partInsIdOffset(i - 1) + partNumInstance(i - 1)
    val bcPartInsIdOffset = spark.sparkContext.broadcast(partInsIdOffset)
    // 3. collect labels of all instances
    // 3.1. create an RDD contains (partInsIdOffset, labels)
    val labelsRdd = oriTrainData.mapPartitionsWithIndex((partId, iterator) => {
      val offsets = bcPartInsIdOffset.value
      val partSize = if (partId + 1 == offsets.length) {
        bcNumTrainData.value - offsets(partId)
      } else {
        offsets(partId + 1) - offsets(partId)
      }
      val labels = new Array[Float](partSize)
      var count = 0
      while (iterator.hasNext) {
        labels(count) = iterator.next().label.toFloat
        count += 1
      }
      require(count == partSize)
      Seq((offsets(partId), labels)).iterator
    }, preservesPartitioning = true)
    // 3.2. collect all labels and put them to corresponding position
    val labels = new Array[Float](bcNumTrainData.value)
    labelsRdd.collect().foreach(part => {
      val offset = part._1
      val partLabels = part._2
      for (i <- partLabels.indices)
        labels(offset + i) = partLabels(i)
    })
    GBDTTrainer.ensureLabel(labels, param.numClass)
    // 3.3. broadcast labels
    val bcLabels = spark.sparkContext.broadcast(labels)
    // 4. generate candidate splits for each feature
    val evenPartitioner = new EvenPartitioner(param.numFeature, param.numWorker)
    val bcFeatureEdges = spark.sparkContext.broadcast(evenPartitioner.partitionEdges())
    // 4.1. create local quantile sketches on each partition
    val localSketchRdd = oriTrainData.mapPartitions(iterator => {
      val numFeature = bcParam.value.numFeature
      val sketches = new Array[(Int, HeapQuantileSketch)](numFeature)
      for (fid <- 0 until numFeature)
        sketches(fid) = (fid, new HeapQuantileSketch())
      while (iterator.hasNext)
        iterator.next().feature.foreachActive((fid, fvalue) => sketches(fid)._2.update(fvalue.toFloat))
      sketches.filter(!_._2.isEmpty).iterator
    })
    // 4.2. repartition to executors evenly and merge as global quantile sketches
    val globalSketchRdd = localSketchRdd.repartitionAndSortWithinPartitions(evenPartitioner)
      .mapPartitions(iterator => {
        val splits = collection.mutable.ArrayBuffer[(Int, Array[Float])]()
        val numSplits = bcParam.value.numSplit
        var curFid = -1
        var curSketch: HeapQuantileSketch = null
        while (iterator.hasNext) {
          val (fid, sketch) = iterator.next()
          if (fid != curFid) {
            if (curFid != -1) {
              splits += ((curFid, Maths.unique(curSketch.getQuantiles(numSplits))))
            }
            curSketch = sketch
            curFid = fid
          } else {
            curSketch.merge(sketch)
          }
        }
        if (curFid != -1)
          splits += ((curFid, Maths.unique(curSketch.getQuantiles(numSplits))))
        splits.iterator
      }, preservesPartitioning = true)
    // 4.3. collect candidates splits for all features
    val splits = new Array[Array[Float]](param.numFeature)
    globalSketchRdd.collect().foreach(s => splits(s._1) = s._2)
    // 4.4. generate feature info and broadcast
    val bcFeatureInfo = spark.sparkContext.broadcast(FeatureInfo(param.numFeature, splits))
    // 5. transpose instances
    // 5.1. map feature values into bin indexes and transpose local data
    val mediumFeatRowRdd = oriTrainData.mapPartitionsWithIndex((partId, iterator) => {
      val numFeature = bcParam.value.numFeature
      val splits = bcFeatureInfo.value.splits
      val insIdLists = new Array[IntArrayList](numFeature)
      val binIdLists = new Array[ByteArrayList](numFeature)
      for (fid <- 0 until numFeature) {
        insIdLists(fid) = new IntArrayList()
        binIdLists(fid) = new ByteArrayList()
      }
      var curInsId = bcPartInsIdOffset.value(partId)
      while (iterator.hasNext) {
        iterator.next().feature.foreachActive((fid, value) => {
          insIdLists(fid).add(curInsId)
          val binId = Maths.indexOf(splits(fid), value.toFloat)
          binIdLists(fid).add((binId + Byte.MinValue).toByte)
        })
        curInsId += 1
      }
      val mediumFeatRows = new ArrayBuffer[(Int, Option[(Array[Int], Array[Byte])])](numFeature)
      for (fid <- 0 until numFeature) {
        val mediumFeatRow =
        if (insIdLists(fid).size() > 0) {
          val featIndices = insIdLists(fid).toIntArray(null)
          val featBins = binIdLists(fid).toByteArray(null)
          (fid, Option((featIndices, featBins)))
        } else {
           (fid, Option.empty)
        }
        mediumFeatRows += mediumFeatRow
      }
      mediumFeatRows.iterator
    })
    // 5.2. repartition feature rows evenly, compact medium feature rows
    // of one feature (from different partition) into one
    val trainData = mediumFeatRowRdd.repartitionAndSortWithinPartitions(evenPartitioner)
      .mapPartitionsWithIndex((partId, iterator) => {
        val featLo = bcFeatureEdges.value(partId)
        val featHi = bcFeatureEdges.value(partId + 1)
        val featureRows = new ArrayBuffer[Option[FeatureRow]](featHi - featLo)
        val partFeatRows = collection.mutable.ArrayBuffer[FeatureRow]()
        var curFid = featLo
        while (iterator.hasNext) {
          val entry = iterator.next()
          val fid = entry._1
          val mediumFeatRow = entry._2 match {
            case Some(pairs) => {
              val indices = pairs._1
              val bins = pairs._2.map(bin => bin.toInt - Byte.MinValue)
              FeatureRow(indices, bins)
            }
            case None => FeatureRow(null, null)
          }
          if (fid != curFid) {
            featureRows += FeatureRow.compact(partFeatRows)
            partFeatRows.clear()
            curFid = fid
            partFeatRows += mediumFeatRow
          } else if (!iterator.hasNext) {
            partFeatRows += mediumFeatRow
            featureRows += FeatureRow.compact(partFeatRows)
          } else {
            partFeatRows += mediumFeatRow
          }
        }
        require(featureRows.size == featHi - featLo)
        featureRows.iterator
      }).persist(StorageLevel.MEMORY_AND_DISK)
    require(trainData.count() == param.numFeature)
    LOG.info(s"Transpose train data cost ${System.currentTimeMillis() - transposeStart} ms, " +
      s"feature edges: [${bcFeatureEdges.value.mkString(", ")}]")

    // 6. initialize an RDD to control each partition
    val partitions = trainData.mapPartitionsWithIndex((partId, _) => {
      var localPartId: Int = 0
      GBDTTrainer.synchronized {
        localPartId = GBDTTrainer.numLocalParts
        GBDTTrainer.numLocalParts += 1
        if (GBDTTrainer.dataInfo == null) {
          GBDTTrainer.dataInfo = DataInfo(bcParam.value, bcNumTrainData.value)
          GBDTTrainer.loss = ObjectiveFactory.getLoss(bcParam.value.lossFunc)
          GBDTTrainer.evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(
            bcParam.value.evalMetrics, GBDTTrainer.loss)
        }
      }
      val featLo = bcFeatureEdges.value(partId)
      val featHi = bcFeatureEdges.value(partId + 1)
      val numSampleFeat = Math.round((featHi - featLo) * bcParam.value.featSampleRatio)
      val sampledFeats = if (numSampleFeat == featHi - featLo) {
        (featLo until featHi).toArray
      } else {
        new Array[Int](numSampleFeat)
      }
      Seq((partId, localPartId, sampledFeats)).iterator
    }, preservesPartitioning = true).persist()

    // 7. make it run
    partitions.foreachPartition(iterator => {
      val partition = iterator.next()
      val partId = partition._1
      LOG.info(s"Partition[$partId] init done")
    })

    // 8. set up transient variable on driver
    this.trainData = trainData
    this.validData = validData
    this.bcNumTrainData = bcNumTrainData
    this.bcNumValidData = bcNumValidData
    this.bcFeatureEdges = bcFeatureEdges
    this.bcFeatureInfo = bcFeatureInfo
    this.bcLabels = bcLabels
    this.partitions = partitions
  }

  def train(): Unit = {
    LOG.info("Start to train GBDT")
    val startTime = System.currentTimeMillis()

    forest = new ArrayBuffer[GBTTree](bcParam.value.numTree)
    phase = GBDTPhase.NEW_TREE
    toBuild = collection.mutable.Map()
    toFind = collection.mutable.Set()
    toSplit = collection.mutable.Map()
    storedNodeHists = collection.mutable.Set()
    partitions.foreachPartition(iterator => {
      val partition = iterator.next()
      val partId = partition._1
      GBDTTrainer.getClass.synchronized {
        if (GBDTTrainer.nodeHists == null)
          GBDTTrainer.nodeHists = collection.mutable.Map()
        GBDTTrainer.nodeHists += partId -> collection.mutable.Map()
        if (GBDTTrainer.workerClock == null)
          GBDTTrainer.workerClock = new Array[Int](GBDTTrainer.numLocalParts)
      }
    })

    while (phase != GBDTPhase.FINISHED) {
      LOG.info(s"******Current phase: $phase******")
      phase match {
        case GBDTPhase.NEW_TREE => createNewTree()
        case GBDTPhase.CHECK_STATUS => checkStatus()
        case GBDTPhase.BUILD_HIST => buildHistogram()
        case GBDTPhase.FIND_SPLIT => findSplit()
        case GBDTPhase.SPLIT_NODE => splitNode()
        case GBDTPhase.FINISH_TREE => {
          finishTree()
          LOG.info(s"${forest.size} tree(s) done, " +
            s"${System.currentTimeMillis() - startTime} ms elapsed")
        }
      }
    }

    LOG.info(s"Train done, ${System.currentTimeMillis() - startTime} ms elapsed")
  }

  def createNewTree(): Unit = {
    LOG.info("------Create new tree------")
    val startTime = System.currentTimeMillis()
    // 1. create new tree
    val tree = new GBTTree(param)
    this.forest += tree
    // 2. sample features and reset position info
    val bcFeatureEdges = this.bcFeatureEdges
    this.partitions.foreachPartition(iterator => {
      val partition = iterator.next()
      val partId = partition._1
      // 2.1. sample features
      val sampledFeat = partition._3
      val featLo = bcFeatureEdges.value(partId)
      val featHi = bcFeatureEdges.value(partId + 1)
      if (sampledFeat.length < featHi - featLo) {
        val temp = (featLo until featHi).toArray
        Maths.shuffle(temp)
        Array.copy(temp, 0, sampledFeat, 0, sampledFeat.length)
        Sorting.quickSort(sampledFeat)
      }
      // 2.2. reset position info
      val localPartId = partition._2
      val dataInfo = GBDTTrainer.dataInfo
      GBDTTrainer.sync(localPartId) {
        dataInfo.resetPosInfo()
      }
    })
    // 3. calc grad pairs
    calcGrad(0)
    // 4. set root as toBuild
    toBuild += 0 -> bcNumTrainData.value
    phase = GBDTPhase.BUILD_HIST
    LOG.info(s"Create new tree cost ${System.currentTimeMillis - startTime} ms")
  }

  def calcGrad(nid: Int): Unit = {
    val bcLabels = this.bcLabels
    partitions.foreachPartition(iterator => {
      val startTime = System.currentTimeMillis()
      val partition = iterator.next()
      val partId = partition._1
      val localPartId = partition._2
      GBDTTrainer.sync(localPartId) {
        val loss = GBDTTrainer.loss
        GBDTTrainer.dataInfo.calcGradPairs(nid, bcLabels.value, loss, bcParam.value)
      }
      LOG.info(s"Part[$partId] calc grad pairs cost ${System.currentTimeMillis() - startTime} ms")
    })
    val sumGradPair = partitions.map(_ => GBDTTrainer.dataInfo.sumGradPair(nid)).take(1)(0)
    forest.last.getNode(nid).setSumGradPair(sumGradPair)
  }

  def checkStatus(): Unit = {
    if (forest.last.size() >= param.maxNodeNum - 1) {
      phase = GBDTPhase.FINISH_TREE
    } else if (toBuild.nonEmpty) {
      phase = GBDTPhase.BUILD_HIST
    } else if (toFind.nonEmpty) {
      phase = GBDTPhase.FIND_SPLIT
    } else if (toSplit.nonEmpty) {
      phase = GBDTPhase.SPLIT_NODE
    } else {
      phase = GBDTPhase.FINISH_TREE
    }
  }

  def buildHistogram(): Unit = {
    LOG.info("------Build histogram------")
    val startTime = System.currentTimeMillis()
    if (toBuild.size == 1) {
      // only one node to build
      val node = toBuild.head
      buildHistogram(node._1)
      toBuild -= node._1
    } else {
      // lighter Child First schema: build histogram for the node which contains less instances
      // and build histogram for its sibling
      val nodes = toBuild.toArray
      for (node <- nodes) {
        val nid = node._1
        if (toBuild.contains(nid)) {
          val siblingNid = Maths.sibling(nid)
          require(toBuild.contains(siblingNid))
          if (param.lighterChildFirst) {
            val mySize = node._2
            val siblingSize = toBuild(siblingNid)
            if (mySize < siblingSize) {
              buildHistogram(nid)
              buildHistogram(siblingNid)
            } else {
              buildHistogram(siblingNid)
              buildHistogram(nid)
            }
          } else {
            buildHistogram(nid min siblingNid)
            buildHistogram(nid max siblingNid)
          }
          toBuild -= nid
          toBuild -= siblingNid
        }
      }
    }
    phase = GBDTPhase.FIND_SPLIT
    LOG.info(s"Build histogram cost ${System.currentTimeMillis - startTime} ms")
  }

  def buildHistogram(nid: Int): Unit = {
    val startTime = System.currentTimeMillis()

    // 1. calculate from subtraction
    var canSubtract = false
    if (nid != 0 && param.histSubtraction) {
      val parentNid = Maths.parent(nid)
      val siblingNid = Maths.sibling(nid)
      if (storedNodeHists.contains(parentNid) && storedNodeHists.contains(siblingNid)) {
        partitions.foreachPartition(iter => {
          val startTime = System.currentTimeMillis()
          val partition = iter.next()
          val partId = partition._1
          val nodeHists = GBDTTrainer.nodeHists(partId)
          val parHist = nodeHists(parentNid)
          val sibHist = nodeHists(siblingNid)
          val numSampledFeats = partition._3.length
          require(parHist.length == numSampledFeats && sibHist.length == numSampledFeats)
          for (i <- 0 until numSampledFeats) {
            if (parHist(i).isDefined && sibHist(i).isDefined) {
              parHist(i).get.subtractBy(sibHist(i).get)
            } else if (parHist(i).isDefined || sibHist(i).isDefined) {
              throw new GBDTException("Histograms of parent's and sibling's do not present together")
            }
          }
          nodeHists -= parentNid
          nodeHists += nid -> parHist
          LOG.info(s"Part[$partId] build histogram for node[$nid] cost " +
            s"${System.currentTimeMillis() - startTime} ms")
        })
        storedNodeHists -= parentNid
        canSubtract = true
      }
    }
    // 2. calculate from data
    if (!canSubtract) {
      val bcFeatureEdges = this.bcFeatureEdges
      val bcFeatureInfo = this.bcFeatureInfo
      val bcSumGradPair = spark.sparkContext.broadcast(
        forest.last.getNode(nid).getSumGradPair)
      partitions.zipPartitions(trainData, preservesPartitioning = true)(
        (iter, featRowIter) => {
          val startTime = System.currentTimeMillis()
          val partition = iter.next()
          val partId = partition._1
          val sampledFeats = partition._3
          val featLo = bcFeatureEdges.value(partId)
          val featureRows = featRowIter.toArray
          val featureInfo = bcFeatureInfo.value
          val dataInfo = GBDTTrainer.dataInfo
          val sumGradPair = bcSumGradPair.value
          val histBuilder = new HistBuilder(bcParam.value)
          val histograms = histBuilder.buildHistograms(sampledFeats, featLo,
            featureRows, featureInfo, dataInfo, nid, sumGradPair)
          GBDTTrainer.nodeHists(partId) += nid -> histograms
          LOG.info(s"Part[$partId] build histogram for node[$nid] " +
            s"cost ${System.currentTimeMillis() - startTime} ms")
          Seq.empty.iterator
        }
      ).count()
    }
    storedNodeHists += nid
    toFind += nid
    LOG.info(s"Build histogram for node[$nid] cost ${System.currentTimeMillis() - startTime} ms")
  }

  def findSplit(): Unit = {
    LOG.info("------Find split------")
    val startTime = System.currentTimeMillis()
    // TODO: Two Side One Pass Split Finding
    toFind.foreach(findSplit); toFind.clear()
    phase = GBDTPhase.SPLIT_NODE
    LOG.info(s"Find split cost ${System.currentTimeMillis - startTime} ms")
  }

  def findSplit(nid: Int): Unit = {
    val startTime = System.currentTimeMillis()
    val nodeGain = this.forest.last.getNode(nid).calcGain(param)
    val bcNodeGain = spark.sparkContext.broadcast(nodeGain)
    val bcSumGradPair = spark.sparkContext.broadcast(
      forest.last.getNode(nid).getSumGradPair)
    val bcFeatureInfo = this.bcFeatureInfo
    val globalBest = partitions.mapPartitions(iter => {
      val startTime = System.currentTimeMillis()
      val partition = iter.next()
      val partId = partition._1
      val sampledFeats = partition._3
      val histograms = GBDTTrainer.nodeHists(partId)(nid)
      val featureInfo = bcFeatureInfo.value
      val sumGradPair = bcSumGradPair.value
      val nodeGain = bcNodeGain.value
      val splitFinder = new SplitFinder(bcParam.value)
      val localBest = splitFinder.findBestSplit(sampledFeats,
        histograms, featureInfo, sumGradPair, nodeGain)
      LOG.info(s"Part[$partId] find best split for node[$nid] cost " +
        s"${System.currentTimeMillis() - startTime} ms, local best split: " +
        s"${localBest.getSplitEntry}")
      if (!bcParam.value.histSubtraction) GBDTTrainer.nodeHists(partId) -= nid
      Seq(localBest).iterator
    }).reduce((s1, s2) => {s1.update(s2); s1})
    LOG.info(s"Find best split for node[$nid] cost ${System.currentTimeMillis() - startTime} ms, " +
      s"global best split: ${globalBest.getSplitEntry}")
    toSplit += nid -> globalBest
  }

  def splitNode(): Unit = {
    LOG.info("------Split node------")
    val startTime = System.currentTimeMillis()
    // TODO: dynamic schema instead of splitting only one node
    var bestNid = -1
    var bestGain = param.minSplitGain
    val leaves = new ArrayBuffer[Int]()
    toSplit.foreach(node => {
      val nid = node._1
      val splitEntry = node._2.getSplitEntry
      if (splitEntry == null || splitEntry.isEmpty
        || splitEntry.getGain <= param.minSplitGain) {
        leaves += nid
      } else if (splitEntry.getGain > bestGain) {
        bestNid = nid
        bestGain = splitEntry.getGain
      }
    })
    if (bestNid != -1) {
      splitNode(bestNid)
      toSplit -= bestNid
    }
    leaves.foreach(leaf => { setNodeAsLeaf(leaf); toSplit -= leaf })
    phase = GBDTPhase.CHECK_STATUS
    LOG.info(s"Split node cost ${System.currentTimeMillis - startTime} ms")
  }

  def splitNode(nid: Int): Unit = {
    val startTime = System.currentTimeMillis()
    val gbtSplit = toSplit(nid)
    val splitEntry = gbtSplit.getSplitEntry
    LOG.info(s"Split node[$nid], split entry: $splitEntry")
    val bcSplitEntry = spark.sparkContext.broadcast(splitEntry)
    val bcSplitFid = spark.sparkContext.broadcast(splitEntry.getFid)
    val bcFeatureEdges = this.bcFeatureEdges
    val bcFeatureInfo = this.bcFeatureInfo
    // 1. responsible executor computes split result
    val splitResult = partitions.zipPartitions(trainData, preservesPartitioning = true)(
      (iter, featRowIter) => {
        val startTime = System.currentTimeMillis()
        val partition = iter.next()
        val partId = partition._1
        val featLo = bcFeatureEdges.value(partId)
        val featHi = bcFeatureEdges.value(partId + 1)
        val splitFid = bcSplitFid.value
        if (featLo <= splitFid && splitFid < featHi) {
          val dataInfo = GBDTTrainer.dataInfo
          // get split entry
          val splitEntry = bcSplitEntry.value
          // get feature row
          val offset = splitFid - featLo
          var cnt = 0
          while (cnt < offset) {
            featRowIter.next()
            cnt += 1
          }
          val featureRow = featRowIter.next().get
          // get candidate splits
          val splits = bcFeatureInfo.value.getSplits(splitFid)
          // generate split result represented in bit set
          val splitResult = dataInfo.getSplitResult(nid, splitEntry, featureRow, splits)
          LOG.info(s"Part[$partId] generate split result cost " +
            s"${System.currentTimeMillis() - startTime} ms")
          Seq(splitResult).iterator
        } else {
          Seq.empty.iterator
        }
      }
    ).collect()
    require(splitResult.length == 1)
    // 2. split node on each executor
    val bcSplitResult = spark.sparkContext.broadcast(splitResult(0))
    val childrenSizes = partitions.mapPartitions(iterator => {
      val startTime = System.currentTimeMillis()
      val partition = iterator.next()
      val partId = partition._1
      val localPartId = partition._2
      GBDTTrainer.sync(localPartId) {
        GBDTTrainer.dataInfo.updatePos(nid, bcSplitResult.value)
      }
      val leftSize = GBDTTrainer.dataInfo.getNodeSize(2 * nid + 1)
      val rightSize = GBDTTrainer.dataInfo.getNodeSize(2 * nid + 2)
      LOG.info(s"Part[$partId] split node[$nid] cost " +
        s"${System.currentTimeMillis() - startTime} ms")
      Seq((leftSize, rightSize)).iterator
    }).reduce((sz1, sz2) => {
      require(sz1._1 == sz2._1 && sz1._2 == sz2._2)
      sz1
    })
    //require(childSizes._1 > 0 && childSizes._2 > 0)
    // 3. prepare for children
    val tree = forest.last
    val node = tree.getNode(nid)
    val leftChild = new GBTNode(2 * nid + 1, node, param.numClass)
    val rightChild = new GBTNode(2 * nid + 2, node, param.numClass)
    leftChild.setSumGradPair(gbtSplit.getLeftGradPair)
    rightChild.setSumGradPair(gbtSplit.getRightGradPair)
    node.setLeftChild(leftChild)
    node.setRightChild(rightChild)
    tree.setNode(2 * nid + 1, leftChild)
    tree.setNode(2 * nid + 2, rightChild)
    if (nid * 2 + 1 < Maths.pow(2, param.maxDepth) - 1) {
      toBuild += (nid * 2 + 1) -> childrenSizes._1
      toBuild += (nid * 2 + 2) -> childrenSizes._2
    } else {
      setNodeAsLeaf(2 * nid + 1)
      setNodeAsLeaf(2 * nid + 2)
    }
    /* for the sake of histogram subtraction
    // 4. if all nodes have values, update instance preds
    val node = forest.last.getNode(nid)
    if (param.numClass == 2) {
      val weight = node.calcWeight(param)
      val bcWeight = spark.sparkContext.broadcast(weight)
      partitions.foreachPartition(iterator => {
        val partition = iterator.next()
        val dataInfo = partition._3
        dataInfo.updatePreds(nid, bcWeight.value)
      })
    } else {
      val weights = node.calcWeights(param)
      val bcWeights = spark.sparkContext.broadcast(weights)
      partitions.foreachPartition(iterator => {
        val partition = iterator.next()
        val dataInfo = partition._3
        dataInfo.updatePreds(nid, bcWeights.value)
      })
    }*/
    LOG.info(s"Split node[$nid] cost ${System.currentTimeMillis() - startTime} ms, " +
      s"left child[${2 * nid + 1}] size[${childrenSizes._1}], " +
      s"right child[${2 * nid + 2}] size[${childrenSizes._2}]")
  }

  def finishTree(): Unit = {
    // 1. set all pending nodes as leaf
    toBuild.foreach(node => setNodeAsLeaf(node._1)); toBuild.clear()
    toFind.foreach(node => setNodeAsLeaf(node)); toFind.clear()
    toSplit.foreach(node => setNodeAsLeaf(node._1)); toSplit.clear()
    partitions.foreach(partition => GBDTTrainer.nodeHists(partition._1).clear()); storedNodeHists.clear()
    storedNodeHists.clear()
    // 2. evaluation on train data
    val bcLabels = this.bcLabels
    val startTime = System.currentTimeMillis()
    val metrics = partitions.mapPartitions(iterator => {
      val partition = iterator.next()
      val partId = partition._1
      val dataInfo = GBDTTrainer.dataInfo
      val evalMetrics = GBDTTrainer.evalMetrics
      val metrics = evalMetrics.map(evalMetric => {
        val metric = evalMetric.eval(dataInfo.predictions, bcLabels.value)
        (evalMetric.getKind, metric)
      })
      val metricMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
      LOG.info(s"Part[$partId] evaluation on train data: $metricMsg")
      Seq(metrics).iterator
    })
    val metricMsg = metrics.take(param.numWorker)(0).map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
    LOG.info(s"Evaluation cost ${System.currentTimeMillis() - startTime} ms")
    LOG.info(s"Evaluation on train data after ${forest.size} tree(s): $metricMsg")
    // 3. TODO: update valid data preds and evaluate
    //
    // 4. continue
    phase = if (forest.size == param.numTree) GBDTPhase.FINISHED else GBDTPhase.NEW_TREE
  }

  def setNodeAsLeaf(nid: Int): Unit = {
    val node = forest.last.getNode(nid)
    node.chgToLeaf()
    if (param.numClass == 2) {
      val weight = node.calcWeight(param)
      LOG.info(s"Set node[$nid] as leaf, weight: $weight")
      val bcWeight = spark.sparkContext.broadcast(weight)
      partitions.foreachPartition(iterator => {
        val partition = iterator.next()
        val localPartId = partition._2
        GBDTTrainer.sync(localPartId) {
          val dataInfo = GBDTTrainer.dataInfo
          dataInfo.updatePreds(nid, bcWeight.value, bcParam.value.learningRate)
        }
      })
    } else {
      val weights = node.calcWeights(param)
      LOG.info(s"Set node[$nid] as leaf, weights: [${weights.mkString(", ")}]")
      val bcWeights = spark.sparkContext.broadcast(weights)
      partitions.foreachPartition(iterator => {
        val partition = iterator.next()
        val localPartId = partition._2
        GBDTTrainer.sync(localPartId) {
          val dataInfo = GBDTTrainer.dataInfo
          dataInfo.updatePreds(nid, bcWeights.value, bcParam.value.learningRate)
        }
      })
    }
  }

}
