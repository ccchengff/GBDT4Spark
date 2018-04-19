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
  @transient private var partitions: RDD[(Int, Array[Int], DataInfo, Loss, Array[EvalMetric])] = _

  // tree info
  @transient private var forest: ArrayBuffer[GBTTree] = _
  @transient private var phase: GBDTPhase = _
  @transient private var toBuild: collection.mutable.Map[Int, Int] = _
  @transient private var toFind: collection.mutable.Set[Int] = _
  @transient private var toSplit: collection.mutable.Map[Int, GBTSplit] = _
  @transient private var nodeHists: collection.mutable.Map[Int, RDD[Option[Histogram]]] = _

  def loadData(input: String, validRatio: Double): Unit = {
    val loadStart = System.currentTimeMillis()
    // 1. load original data, split into train data and valid data
    val dim = param.numFeature
    val dataset = spark.sparkContext.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => DataLoader.parseLibsvm(line, dim))
      .randomSplit(Array[Double](1.0 - validRatio, validRatio))
    val oriTrainData = dataset(0).persist(StorageLevel.MEMORY_AND_DISK)
    val validData = dataset(1).persist(StorageLevel.MEMORY_AND_DISK)
    val bcNumTrainData = spark.sparkContext.broadcast(oriTrainData.count().toInt)
    val bcNumValidData = spark.sparkContext.broadcast(validData.count().toInt)
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
    // TODO: support discrete value features
    // 4.1. create local quantile sketches on each partition
    val qSketchRdd = oriTrainData.mapPartitions(iterator => {
      val numFeature = bcParam.value.numFeature
      val qSketches = new Array[HeapQuantileSketch](numFeature)
      for (fid <- 0 until numFeature)
        qSketches(fid) = new HeapQuantileSketch()
      while (iterator.hasNext)
        iterator.next().feature.foreachActive((fid, value) => qSketches(fid).update(value.toFloat))
      Seq(qSketches).iterator
    })
    // 4.2. merge as global quantile sketches and query quantiles as candidate split points
    val splits = qSketchRdd.aggregate(new Array[HeapQuantileSketch](bcParam.value.numFeature))(
      seqOp = (c, v) => v,
      combOp = (c1, c2) => {
        val numFeature = bcParam.value.numFeature
        for (fid <- 0 until numFeature) {
          if (c1(fid) == null) c1(fid) = c2(fid)
          else c1(fid).merge(c2(fid))
        }
        c1
      }
    ).map(_.getQuantiles(bcParam.value.numSplit))
    // 4.3. generate feature info
    val isCategorical = Array.fill[Boolean](bcParam.value.numFeature)(false)
    val featureInfo = FeatureInfo(isCategorical, splits)
    // 4.4. broadcast feature info
    val bcFeatureInfo = spark.sparkContext.broadcast(featureInfo)
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
    val evenPartitioner = new EvenPartitioner(param.numFeature, param.numWorker)
    val bcFeatureEdges = spark.sparkContext.broadcast(evenPartitioner.partitionEdges())
    val trainData = mediumFeatRowRdd.repartitionAndSortWithinPartitions(evenPartitioner)
      .mapPartitionsWithIndex((partId, iterator) => {
        val featLo = bcFeatureEdges.value(partId)
        val featHi = bcFeatureEdges.value(partId + 1)
        val featureRows = new ArrayBuffer[Option[FeatureRow]](featHi - featLo)
        //val partFeatRows = new util.ArrayList[FeatureRow]()
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
      val featLo = bcFeatureEdges.value(partId)
      val featHi = bcFeatureEdges.value(partId + 1)
      val numSampleFeat = Math.round((featHi - featLo) * bcParam.value.featSampleRatio)
      val sampledFeats = if (numSampleFeat == featHi - featLo) {
        (featLo until featHi).toArray
      } else {
        new Array[Int](numSampleFeat)
      }
      val dataInfo = DataInfo(bcParam.value, bcNumTrainData.value)
      val loss = ObjectiveFactory.getLoss(bcParam.value.lossFunc)
      val evalMetrics = ObjectiveFactory.getEvalMetricsOrDefault(bcParam.value.evalMetrics, loss)
      Seq((partId, sampledFeats, dataInfo, loss, evalMetrics)).iterator
    }, preservesPartitioning = true).persist()

    // 7. make it run
    partitions.foreachPartition(iterator => {
      val partId = iterator.next()._1
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
    nodeHists = collection.mutable.Map()

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
      val sampledFeat = partition._2
      val featLo = bcFeatureEdges.value(partId)
      val featHi = bcFeatureEdges.value(partId + 1)
      if (sampledFeat.length < featHi - featLo) {
        val temp = (featLo until featHi).toArray
        Maths.shuffle(temp)
        Array.copy(temp, 0, sampledFeat, 0, sampledFeat.length)
        Sorting.quickSort(sampledFeat)
      }
      // 2.2. reset position info
      val dataInfo = partition._3
      dataInfo.resetPosInfo()
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
    val sumGradPair = partitions.mapPartitions(iterator => {
      val partition = iterator.next()
      val dataInfo = partition._3
      val loss = partition._4
      val sumGradPair = dataInfo.calcGradPairs(nid, bcLabels.value, loss, bcParam.value)
      Seq(sumGradPair).iterator
    }, preservesPartitioning = true)
      .treeReduce((gp1, gp2) => {gp1.plusBy(gp2); gp1})
    sumGradPair.timesBy(1.0f / partitions.getNumPartitions)

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
          val mySize = node._2
          val siblingSize = toBuild(siblingNid)
          if (mySize < siblingSize) {
            buildHistogram(nid)
            buildHistogram(siblingNid)
          } else {
            buildHistogram(siblingNid)
            buildHistogram(nid)
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
    var resultRdd: RDD[Option[Histogram]] = null

    // 1. calculate from subtraction
    if (nid != 0) {
      val parentNid = Maths.parent(nid)
      val siblingNid = Maths.sibling(nid)
      if (nodeHists.contains(parentNid) && nodeHists.contains(siblingNid)) {
        val parHistRdd = nodeHists(parentNid)
        val sibHistRdd = nodeHists(siblingNid)
        resultRdd = partitions.zipPartitions(parHistRdd, sibHistRdd, preservesPartitioning = true)(
          (iter, parIter, sibIter) => {
            val startTime = System.currentTimeMillis()
            val partition = iter.next()
            val partId = partition._1
            val numSampledFeats = partition._2.length
            val histograms = new ArrayBuffer[Option[Histogram]](numSampledFeats)
            while (parIter.hasNext && sibIter.hasNext) {
              val hist = (parIter.next(), sibIter.next()) match {
                case (Some(parHist), Some(sibHist)) => Option(parHist.subtract(sibHist))
                case (None, None) => Option.empty
                case (Some(_), None) | (None, Some(_)) => throw new GBDTException(
                  "Histograms of parent's and sibling's do not present together")
              }
              histograms += hist
            }
            require(!parIter.hasNext && !sibIter.hasNext)
            LOG.info(s"Part[$partId] build histogram for node[$nid] cost " +
              s"${System.currentTimeMillis() - startTime} ms")
            histograms.iterator
          }
        ).persist()
        resultRdd.count()
        parHistRdd.unpersist()
        nodeHists -= parentNid
      }
    }
    // 2. calculate from data
    if (resultRdd == null) {
      val bcFeatureEdges = this.bcFeatureEdges
      val bcFeatureInfo = this.bcFeatureInfo
      val bcSumGradPair = spark.sparkContext.broadcast(
        forest.last.getNode(nid).getSumGradPair)
      resultRdd = partitions.zipPartitions(trainData, preservesPartitioning = true)(
        (iter, featRowIter) => {
          val startTime = System.currentTimeMillis()
          val partition = iter.next()
          val partId = partition._1
          // 1. get sampled feature rows
          val sampledFeats = partition._2
          val numSampledFeat = sampledFeats.length
          val sampledFeatRows = new Array[FeatureRow](numSampledFeat)
          var curFid = bcFeatureEdges.value(partId)
          var cnt = 0
          while (featRowIter.hasNext && cnt < numSampledFeat) {
            val featRow = featRowIter.next()
            if (curFid == sampledFeats(cnt)) {
              sampledFeatRows(cnt) = featRow match {
                case Some(row) => row
                case None => null
              }
              cnt += 1
            }
            curFid += 1
          }
          // 2. get instance position info
          val targetNid = nid
          val dataInfo = partition._3
          val nodeStart = dataInfo.nodePosStart(targetNid)
          val nodeEnd = dataInfo.nodePosEnd(targetNid)
          val insPos = dataInfo.insPos
          // 3. get grad pairs and sum of grad pairs
          val gradPairs = dataInfo.gradPairs
          val sumGradPair = bcSumGradPair.value
          // 3. get default bins
          val defaultBins = bcFeatureInfo.value.defaultBins
          val histBuilder = new HistBuilder(bcParam.value)
          val histograms = histBuilder.buildHistograms(sampledFeatRows,
            nodeStart, nodeEnd, insPos, gradPairs, sumGradPair, defaultBins)
            .map(hist => Option(hist))
          LOG.info(s"Part[$partId] build histogram for node[$nid] " +
            s"cost ${System.currentTimeMillis() - startTime} ms")
          histograms.iterator
        }
      ).persist()
      resultRdd.count()
    }
    nodeHists += nid -> resultRdd
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
    val nodeHist = this.nodeHists(nid)
    val bcSumGradPair = spark.sparkContext.broadcast(
      forest.last.getNode(nid).getSumGradPair)
    val bcFeatureInfo = this.bcFeatureInfo
    val globalBest = partitions.zipPartitions(nodeHist, preservesPartitioning = true)(
      (iter, histIter) => {
        val startTime = System.currentTimeMillis()
        val partition = iter.next()
        val partId = partition._1
        val sampledFeats = partition._2
        val featureInfo = bcFeatureInfo.value
        val sumGradPair = bcSumGradPair.value
        val nodeGain = bcNodeGain.value
        val splitFinder = new SplitFinder(bcParam.value)
        val localBest = new GBTSplit()
        var cnt = 0
        while (histIter.hasNext) {
          histIter.next() match {
            case Some(hist) => {
              val fid = sampledFeats(cnt)
              val isCategorical = featureInfo.isCategorical(fid)
              val splits = featureInfo.getSplits(fid)
              val defaultBin = featureInfo.getDefaultBin(fid)
              val gbtSplit = splitFinder.findBestSplitOfOneFeature(
                fid, isCategorical, splits, defaultBin, hist, sumGradPair, nodeGain)
              localBest.update(gbtSplit)
            }
            case None =>
          }
          cnt += 1
        }
        LOG.info(s"Part[$partId] find best split for node[$nid] cost " +
          s"${System.currentTimeMillis() - startTime} ms, local best split: " +
          s"${localBest.getSplitEntry}")
        require(cnt == sampledFeats.length)
        Seq(localBest).iterator
      }
    ).reduce((s1, s2) => {s1.update(s2); s1})
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
      if (splitEntry.isEmpty || splitEntry.getGain <= param.minSplitGain) {
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
    //val gbtSplit = toSplit.get(nid)
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
          val dataInfo = partition._3
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
    val childSizes = partitions.mapPartitions(iterator => {
      val startTime = System.currentTimeMillis()
      val partition = iterator.next()
      val partId = partition._1
      val dataInfo = partition._3
      val childrenSizes = dataInfo.updatePos(nid, bcSplitResult.value)
      LOG.info(s"Part[$partId] split node[$nid] cost " +
        s"${System.currentTimeMillis() - startTime} ms")
      Seq(childrenSizes).iterator
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
      toBuild += (nid * 2 + 1) -> childSizes._1
      toBuild += (nid * 2 + 2) -> childSizes._2
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
      s"left child[${2 * nid + 1}] size[${childSizes._1}], " +
      s"right child[${2 * nid + 2}] size[${childSizes._2}]")
  }

  def finishTree(): Unit = {
    // 1. set all pending nodes as leaf
    toBuild.foreach(node => setNodeAsLeaf(node._1)); toBuild.clear()
    toFind.foreach(node => setNodeAsLeaf(node)); toFind.clear()
    toSplit.foreach(node => setNodeAsLeaf(node._1)); toSplit.clear()
    nodeHists.foreach(node => node._2.unpersist()); nodeHists.clear()
    // 2. evaluation on train data
    val bcLabels = this.bcLabels
    val metrics = partitions.mapPartitions(iterator => {
      val partition = iterator.next()
      val partId = partition._1
      val dataInfo = partition._3
      val evalMetrics = partition._5
      val metrics = evalMetrics.map(evalMetric => {
        val metric = evalMetric.eval(dataInfo.predictions, bcLabels.value)
        (evalMetric.getKind, metric)
      })
      val metricMsg = metrics.map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
      LOG.info(s"Part[$partId] evaluation on train data: $metricMsg")
      Seq(metrics).iterator
    })
    metrics.count()
    val metricMsg = metrics.take(1)(0).map(metric => s"${metric._1}[${metric._2}]").mkString(", ")
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
        val dataInfo = partition._3
        dataInfo.updatePreds(nid, bcWeight.value, bcParam.value.learningRate)
      })
    } else {
      val weights = node.calcWeights(param)
      LOG.info(s"Set node[$nid] as leaf, weights: [${weights.mkString(", ")}]")
      val bcWeights = spark.sparkContext.broadcast(weights)
      partitions.foreachPartition(iterator => {
        val partition = iterator.next()
        val dataInfo = partition._3
        dataInfo.updatePreds(nid, bcWeights.value, bcParam.value.learningRate)
      })
    }
  }

}
