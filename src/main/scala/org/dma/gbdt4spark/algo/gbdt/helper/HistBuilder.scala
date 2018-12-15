package org.dma.gbdt4spark.algo.gbdt.helper

import java.util.concurrent.{Callable, Executors}

import org.dma.gbdt4spark.algo.gbdt.dataset.{Dataset, Partition}
import org.dma.gbdt4spark.algo.gbdt.histogram.{GradPair, Histogram}
import org.dma.gbdt4spark.algo.gbdt.metadata.{FeatureInfo, InstanceInfo}
import org.dma.gbdt4spark.tree.param.GBDTParam

object HistBuilder {

  private val MIN_INSTANCE_PER_THREAD = 100000

  // for root node
  def sparseBuild(param: GBDTParam, partition: Partition[Int, Int],
                  insIdOffset: Int, instanceInfo: InstanceInfo,
                  isFeatUsed: Array[Boolean], histograms: Array[Histogram]): Unit = {
    val num = partition.size
    val gradients = instanceInfo.gradients
    val hessians = instanceInfo.hessians
    val fids = partition.indices
    val bins = partition.values
    val indexEnds = partition.indexEnds
    if (param.numClass == 2) {
      var indexStart = 0
      for (i <- 0 until num) {
        val insId = i + insIdOffset
        val grad = gradients(insId)
        val hess = hessians(insId)
        val indexEnd = indexEnds(i)
        for (j <- indexStart until indexEnd) {
          if (isFeatUsed(fids(j))) {
            histograms(fids(j)).accumulate(bins(j), grad, hess)
          }
        }
        indexStart = indexEnd
      }
    } else if (!param.fullHessian) {
      var indexStart = 0
      for (i <- 0 until num) {
        val insId = i + insIdOffset
        val indexEnd = indexEnds(i)
        for (j <- indexStart until indexEnd) {
          if (isFeatUsed(fids(j))) {
            histograms(fids(j)).accumulate(bins(j),
              gradients, hessians, insId * param.numClass)
          }
        }
        indexStart = indexEnd
      }
    } else {
      throw new UnsupportedOperationException("Full hessian not supported")
    }
  }

  // for nodes with few instances
  def sparseBuild(param: GBDTParam, dataset: Dataset[Int, Int],
                  instanceInfo: InstanceInfo, insIds: Array[Int], start: Int, end: Int,
                  isFeatUsed: Array[Boolean], histograms: Array[Histogram]): Unit = {
    val gradients = instanceInfo.gradients
    val hessians = instanceInfo.hessians
    val insLayouts = dataset.insLayouts
    val partitions = dataset.partitions
    val partOffsets = dataset.partOffsets
    if (param.numClass == 2) {
      for (i <- start until end) {
        val insId = insIds(i)
        val grad = gradients(insId)
        val hess = hessians(insId)
        val partId = insLayouts(insId)
        val fids = partitions(partId).indices
        val bins = partitions(partId).values
        val partInsId = insId - partOffsets(partId)
        val indexStart = if (partInsId == 0) 0 else partitions(partId).indexEnds(partInsId - 1)
        val indexEnd = partitions(partId).indexEnds(partInsId)
        for (j <- indexStart until indexEnd) {
          if (isFeatUsed(fids(j))) {
            histograms(fids(j)).accumulate(bins(j), grad, hess)
          }
        }
      }
    } else if (!param.fullHessian) {
      for (i <- start until end) {
        val insId = insIds(i)
        val partId = insLayouts(insId)
        val fids = partitions(partId).indices
        val bins = partitions(partId).values
        val partInsId = insId - partOffsets(partId)
        val indexStart = if (partInsId == 0) 0 else partitions(partId).indexEnds(partInsId - 1)
        val indexEnd = partitions(partId).indexEnds(partInsId)
        for (j <- indexStart until indexEnd) {
          if (isFeatUsed(fids(j))) {
            histograms(fids(j)).accumulate(bins(j),
              gradients, hessians, insId * param.numClass)
          }
        }
      }
    } else {
      throw new UnsupportedOperationException("Full hessian not supported")
    }
  }

  def fillDefaultBins(param: GBDTParam, featureInfo: FeatureInfo,
                      sumGradPair: GradPair, histograms: Array[Histogram]): Unit = {
    for (fid <- histograms.indices) {
      if (histograms(fid) != null) {
        val taken = histograms(fid).sum()
        val remain = sumGradPair.subtract(taken)
        val defaultBin = featureInfo.getDefaultBin(fid)
        histograms(fid).accumulate(defaultBin, remain)
      }
    }
  }

  def apply(param: GBDTParam, dataset: Dataset[Int, Int],
            instanceInfo: InstanceInfo, featureInfo: FeatureInfo): HistBuilder =
    new HistBuilder(param, dataset, instanceInfo, featureInfo)

}

import HistBuilder._

class HistBuilder(param: GBDTParam, dataset: Dataset[Int, Int],
                  instanceInfo: InstanceInfo, featureInfo: FeatureInfo) {

  private val threadPool = Executors.newFixedThreadPool(param.numThread)

  def buildHistForRoot(isFeatUsed: Array[Boolean], sumGradPair: GradPair): Array[Histogram] = {
    val numFeat = featureInfo.numFeature
    val histograms = new Array[Histogram](featureInfo.numFeature)
    for (fid <- 0 until numFeat) {
      if (isFeatUsed(fid))
        histograms(fid) = new Histogram(featureInfo.getNumBin(fid),
          param.numClass, param.fullHessian)
    }
    if (param.numThread == 1 || dataset.size < MIN_INSTANCE_PER_THREAD) {
      for (partId <- 0 until dataset.numPartition) {
        val partition = dataset.partitions(partId)
        val insIdOffset = dataset.partOffsets(partId)
        sparseBuild(param, partition, insIdOffset, instanceInfo, isFeatUsed, histograms)
      }
    } else {
      val futures = (0 until dataset.numPartition).map(partId => {
        val partition = dataset.partitions(partId)
        val insIdOffset = dataset.partOffsets(partId)
        threadPool.submit(new Runnable {
          override def run(): Unit = sparseBuild(param, partition,
            insIdOffset, instanceInfo, isFeatUsed, histograms)
        })
      })
      futures.foreach(_.get)
    }
    fillDefaultBins(param, featureInfo, sumGradPair, histograms)
    histograms
  }

  def buildHistForNodes(nids: Seq[Int], isFeatUsed: Array[Boolean], sumGradPairs: Seq[GradPair],
                        subtracts: Seq[Array[Histogram]]): Array[Array[Histogram]] = {
    if (param.numThread == 1 || nids.length == 1) {
      nids.indices.map(i =>
        buildHistForNode(nids(i), instanceInfo, featureInfo, isFeatUsed,
          sumGradPairs(i), subtracts(i))
      ).toArray
    } else {
      val futures = nids.indices.map(i => threadPool.submit(new Callable[Array[Histogram]] {
        override def call(): Array[Histogram] = buildHistForNode(nids(i),
          instanceInfo, featureInfo, isFeatUsed, sumGradPairs(i), subtracts(i))
      })).toArray
      futures.map(_.get)
    }
  }

  def buildHistForNode(nid: Int, instanceInfo: InstanceInfo, featureInfo: FeatureInfo,
                       isFeatUsed: Array[Boolean], sumGradPair: GradPair,
                       subtract: Array[Histogram] = null): Array[Histogram] = {
    val numFeat = featureInfo.numFeature
    val histograms = new Array[Histogram](featureInfo.numFeature)
    for (fid <- 0 until numFeat) {
      if (isFeatUsed(fid))
        histograms(fid) = new Histogram(featureInfo.getNumBin(fid),
          param.numClass, param.fullHessian)
    }
    sparseBuild(param, dataset, instanceInfo, instanceInfo.nodeToIns,
      instanceInfo.getNodePosStart(nid), instanceInfo.getNodePosEnd(nid),
      isFeatUsed, histograms)
    fillDefaultBins(param, featureInfo, sumGradPair, histograms)
    if (subtract != null) {
      for (fid <- 0 until numFeat) {
        if (isFeatUsed(fid))
          subtract(fid).subtractBy(histograms(fid))
      }
    }
    histograms
  }

  def shutdown(): Unit = {
    threadPool.shutdown()
  }

}
