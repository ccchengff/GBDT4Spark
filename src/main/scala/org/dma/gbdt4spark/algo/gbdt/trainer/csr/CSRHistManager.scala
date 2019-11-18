package org.dma.gbdt4spark.algo.gbdt.trainer.csr

import java.util.concurrent.{Callable, ExecutorService}

import org.dma.gbdt4spark.algo.gbdt.histogram.{GradPair, Histogram}
import org.dma.gbdt4spark.algo.gbdt.metadata.{FeatureInfo, InstanceInfo}
import org.dma.gbdt4spark.tree.param.GBDTParam
import java.{util => ju}

import org.dma.gbdt4spark.algo.gbdt.helper.HistManager.NodeHist
import org.dma.gbdt4spark.util.{ConcurrentUtil, Maths}

object CSRHistManager {

  private val MIN_COLUMN_PER_THREAD = 10
  private val MAX_COLUMN_PER_THREAD = 1000

  def sparseBuildRoot(param: GBDTParam, columns: Array[Column], start: Int, end: Int,
                      insInfo: InstanceInfo, histograms: NodeHist): Unit = {
    val gradients = insInfo.gradients
    val hessians = insInfo.hessians

    if (param.numClass == 2) {
      for (fid <- start until end) {
        val histogram = histograms(fid)
        if (histogram != null) {
          val indices = columns(fid).indices
          val bins = columns(fid).bins
          for (i <- indices.indices) {
            val insId = indices(i)
            val binId = bins(i)
            val grad = gradients(insId)
            val hess = hessians(insId)
            histogram.accumulate(binId, grad, hess)
          }
        }
      }
    } else if (!param.fullHessian) {
      for (fid <- start until end) {
        val histogram = histograms(fid)
        if (histogram != null) {
          val indices = columns(fid).indices
          val bins = columns(fid).bins
          for (i <- indices.indices) {
            val insId = indices(i)
            val binId = bins(i)
            histogram.accumulate(binId, gradients, hessians, insId * param.numClass)
          }
        }
      }
    }
  }

//  def sparseBuild(param: GBDTParam, columns: Array[Column], start: Int, end: Int,
//                  insInfo: InstanceInfo, insToNode: Array[(Int, Int)], histograms: NodeHist): Unit = {
//    val gradients = insInfo.gradients
//    val hessians = insInfo.hessians
//    val numIns = insToNode.length
//
//    require(param.numClass == 2)
//    for (fid <- start until end) {
//      val histogram = histograms(fid)
//      if (histogram != null) {
//        val indices = columns(fid).indices
//        val bins = columns(fid).bins
//        val colSize = indices.length
//
//        var i = 0
//        var j = 0
//        while (i < colSize && j < numIns) {
//          val insId1 = indices(i)
//          val insId2 = insToNode(j)._1
//          if (insId1 < insId2) {
//            i += 1
//          } else if (insId1 > insId2) {
//            j += 1
//          } else {
//            val binId = bins(i)
//            histogram.accumulate(binId, gradients(insId1), hessians(insId1))
//            i += 1
//            j += 1
//          }
//        }
//      }
//    }
//  }

  // for nodes with few instances
  def sparseBuild(param: GBDTParam, columns: Array[Column], start: Int, end: Int,
                  nid: Int, insInfo: CSRInstanceInfo, histograms: NodeHist): Unit = {
    val gradients = insInfo.gradients
    val hessians = insInfo.hessians

    val nodeToIns = insInfo.nodeToIns
    val nodeStart = insInfo.getNodePosStart(nid)
    val nodeEnd = insInfo.getNodePosEnd(nid)
    val insPos = insInfo.insPos

    if (param.numClass == 2) {
      for (fid <- start until end) {
        val histogram = histograms(fid)
        if (histogram != null) {
          val indices = columns(fid).indices
          val bins = columns(fid).bins
          val colSize = indices.length

          if (colSize > 100 * (nodeEnd - nodeStart + 1)) {
            var offset = 0
            for (posId <- nodeStart to nodeEnd) {
              val insId = nodeToIns(posId)
              val t = ju.Arrays.binarySearch(indices, offset, colSize, posId)
              if (t >= 0) {
                val binId = bins(t)
                val grad = gradients(insId)
                val hess = hessians(insId)
                histogram.accumulate(binId, grad, hess)
                offset = t + 1
              }
            }
          } else {
            for (i <- 0 until colSize) {
              val insId = indices(i)
              if (insPos(insId) == nid) {
//              val posId = insPos(insId)
//              if (nodeStart <= posId && posId <= nodeEnd) {
                val binId = bins(i)
                val grad = gradients(insId)
                val hess = hessians(insId)
                histogram.accumulate(binId, grad, hess)
              }
            }
          }
        }
      }
    } else if (!param.fullHessian) {
      for (fid <- start until end) {
        val histogram = histograms(fid)
        if (histogram != null) {
          val indices = columns(fid).indices
          val bins = columns(fid).bins
          val colSize = indices.length

          if (colSize > 100 * (nodeEnd - nodeStart + 1)) {
            var offset = 0
            for (posId <- nodeStart to nodeEnd) {
              val insId = nodeToIns(posId)
              val t = ju.Arrays.binarySearch(indices, offset, colSize, posId)
              if (t >= 0) {
                val binId = bins(t)
                histogram.accumulate(binId, gradients, hessians, insId * param.numClass)
                offset = t + 1
              }
            }
          } else {
            for (i <- 0 until colSize) {
              val insId = indices(i)
              if (insPos(insId) == nid) {
                //              val posId = insPos(insId)
                //              if (nodeStart <= posId && posId <= nodeEnd) {
                val binId = bins(i)
                histogram.accumulate(binId, gradients, hessians, insId * param.numClass)
              }
            }
          }
        }
      }
    }

  }

  def fillDefaultBins(param: GBDTParam, featureInfo: FeatureInfo,
                      sumGradPair: GradPair, histograms: NodeHist): Unit = {
    for (fid <- histograms.indices) {
      if (histograms(fid) != null) {
        val taken = histograms(fid).sum()
        val remain = sumGradPair.subtract(taken)
        val defaultBin = featureInfo.getDefaultBin(fid)
        histograms(fid).accumulate(defaultBin, remain)
      }
    }
  }

  private def allocNodeHist(param: GBDTParam, featureInfo: FeatureInfo,
                            isFeatUsed : Array[Boolean]): NodeHist = {
    val numFeat = featureInfo.numFeature
    val histograms = Array.ofDim[Histogram](numFeat)
    for (fid <- 0 until numFeat) {
      if (isFeatUsed(fid))
        histograms(fid) = new Histogram(featureInfo.getNumBin(fid),
          param.numClass, param.fullHessian)
    }
    histograms
  }

  def apply(param: GBDTParam, featureInfo: FeatureInfo): CSRHistManager = new CSRHistManager(param, featureInfo)

}

import CSRHistManager._
class CSRHistManager(param: GBDTParam, featureInfo: FeatureInfo) {
  private[gbdt] var isFeatUsed : Array[Boolean] = _
  private[gbdt] val nodeGradPairs = new Array[GradPair](Maths.pow(2, param.maxDepth + 1) - 1)
  private[gbdt] var nodeHists = new Array[NodeHist](Maths.pow(2, param.maxDepth) - 1)
  private[gbdt] var histStore = new Array[NodeHist](Maths.pow(2, param.maxDepth) - 1)
  private[gbdt] var availHist = 0

  private class NodeHistPool(capacity: Int) {
    private val pool = Array.ofDim[NodeHist](capacity)
    private var numHist = 0
    private var numAcquired = 0

    private[gbdt] def acquire: NodeHist = {
      this.synchronized {
        if (numHist == numAcquired) {
          require(numHist < pool.length)
          pool(numHist) = getOrAllocSync(sync = true)
          numHist += 1
        }
        var i = 0
        while (i < numHist && pool(i) == null) i += 1
        numAcquired += 1
        val nodeHist = pool(i)
        pool(i) = null
        nodeHist
      }
    }

    private[gbdt] def release(nodeHist: NodeHist): Unit = {
      this.synchronized {
        require(numHist > 0)
        var i = 0
        while (i < numHist && pool(i) != null) i += 1
        pool(i) = nodeHist
        numAcquired -= 1
      }
    }

    private[gbdt] def result: NodeHist = {
      require(numHist > 0 && numAcquired == 0)
      val res = pool.head
      for (i <- 1 until numHist) {
        val one = pool(i)
        for (fid <- isFeatUsed.indices)
          if (res(fid) != null)
            res(fid).plusBy(one(fid))
        releaseSync(one, sync = true)
      }
      res
    }
  }

  def buildHistForRoot(dataset: CSRDataset, insInfo: InstanceInfo, threadPool: ExecutorService = null): Unit = {
    val histograms = if (param.numThread == 1 || dataset.numColumn < MIN_COLUMN_PER_THREAD) {
      val nodeHist = getOrAllocSync()
      sparseBuildRoot(param, dataset.columns, 0, dataset.numColumn, insInfo, nodeHist)
      nodeHist
    } else {
      val histPool = new NodeHistPool(param.numThread)
      val batchSize = MIN_COLUMN_PER_THREAD max (MAX_COLUMN_PER_THREAD min
        Maths.idivCeil(dataset.numColumn, param.numThread))
      val thread = (start: Int, end: Int) => {
        val nodeHist = histPool.acquire
        nodeHist.synchronized {
          sparseBuildRoot(param, dataset.columns, start, end, insInfo, nodeHist)
        }
        histPool.release(nodeHist)
      }
      val futures = ConcurrentUtil.rangeParallel(thread, 0, dataset.numColumn,
        threadPool, batchSize = batchSize)
      futures.foreach(_.get)
      histPool.result
    }
    fillDefaultBins(param, featureInfo, nodeGradPairs(0), histograms)
    setNodeHist(0, histograms)
  }

  def buildHistForNodes(nids: Seq[Int], dataset: CSRDataset, insInfo: CSRInstanceInfo,
                        subtracts: Seq[Boolean], threadPool: ExecutorService = null): Unit = {
//    val numIns = nids.map(insInfo.getNodeSize).sum
//    var insToNode = new Array[(Int, Int)](numIns)
//    val nodeToIns = insInfo.nodeToIns
//    var offset = 0
//    nids.foreach(nid => {
//      val nodeStart = insInfo.getNodePosStart(nid)
//      val nodeEnd = insInfo.getNodePosEnd(nid)
//      for (posId <- nodeStart to nodeEnd) {
//        val insId = nodeToIns(posId)
//        insToNode(offset + posId - nodeStart) = (insId, nid)
//      }
//      offset += nodeEnd - nodeStart + 1
//    })
//    insToNode = insToNode.sortBy(_._1)

    if (param.numThread == 1 || nids.length == 1) {
      (nids, subtracts).zipped.foreach {
        case (nid, subtract) => buildHistForNode(nid, dataset, insInfo, subtract)
      }
    } else {
      val futures = (nids, subtracts).zipped.map {
        case (nid, subtract) => threadPool.submit(new Callable[Unit] {
          override def call(): Unit =
            buildHistForNode(nid, dataset, insInfo, subtract, sync = true)
        })
      }
      futures.foreach(_.get)
    }
  }

  def buildHistForNode(nid: Int, dataset: CSRDataset, insInfo: CSRInstanceInfo,
                       subtract: Boolean = false, sync: Boolean = false): Unit = {
    val nodeHist = getOrAllocSync(sync = sync)
    sparseBuild(param, dataset.columns, 0, dataset.numColumn, nid, insInfo, nodeHist)
    fillDefaultBins(param, featureInfo, nodeGradPairs(nid), nodeHist)
    setNodeHist(nid, nodeHist)
    if (subtract) {
      histSubtract(nid, Maths.sibling(nid), Maths.parent(nid))
    } else {
      removeNodeHist(Maths.parent(nid), sync = sync)
    }
  }

  def getGradPair(nid: Int): GradPair = nodeGradPairs(nid)

  def setGradPair(nid: Int, gp: GradPair): Unit = nodeGradPairs(nid) = gp

  def getNodeHist(nid: Int): NodeHist = nodeHists(nid)

  def setNodeHist(nid: Int, nodeHist: NodeHist): Unit = nodeHists(nid) = nodeHist

  def removeNodeHist(nid: Int, sync: Boolean = false): Unit  = {
    val nodeHist = nodeHists(nid)
    if (nodeHist != null) {
      nodeHists(nid) = null
      releaseSync(nodeHist, sync = sync)
    }
  }

  def histSubtract(nid: Int, sibling: Int, parent: Int): Unit = {
    val mined = nodeHists(parent)
    val miner = nodeHists(nid)
    for (fid <- isFeatUsed.indices)
      if (mined(fid) != null)
        mined(fid).subtractBy(miner(fid))
    nodeHists(sibling) = mined
    nodeHists(parent) = null
  }

  private def getOrAllocSync(sync: Boolean = false): NodeHist = {
    def doGetOrAlloc(): NodeHist = {
      if (availHist == 0) {
        allocNodeHist(param, featureInfo, isFeatUsed)
      } else {
        val res = histStore(availHist - 1)
        histStore(availHist - 1) = null
        availHist -= 1
        res
      }
    }

    if (sync) this.synchronized(doGetOrAlloc())
    else doGetOrAlloc()
  }

  private def releaseSync(nodeHist: NodeHist, sync: Boolean = false): Unit = {
    def doRelease(): Unit = {
      for (hist <- nodeHist)
        if (hist != null)
          hist.clear()
      histStore(availHist) = nodeHist
      availHist += 1
    }

    if (sync) this.synchronized(doRelease())
    else doRelease()
  }

  def reset(isFeatUsed: Array[Boolean]): Unit = {
    require(isFeatUsed.length == featureInfo.numFeature)
    this.isFeatUsed = isFeatUsed
    for (i <- histStore.indices)
      histStore(i) = null
    availHist = 0
    System.gc()
  }

}
