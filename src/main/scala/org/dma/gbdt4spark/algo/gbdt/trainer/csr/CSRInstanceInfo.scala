package org.dma.gbdt4spark.algo.gbdt.trainer.csr

import java.util.concurrent.ExecutorService

import org.dma.gbdt4spark.algo.gbdt.metadata.InstanceInfo
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{ConcurrentUtil, Maths, RangeBitSet}

object CSRInstanceInfo {

  def apply(param: GBDTParam, numData: Int): CSRInstanceInfo = {
    val size = if (param.numClass == 2) numData else param.numClass * numData
    val predictions = new Array[Float](size)
    val weights = Array.fill[Float](numData)(1.0f)
    val gradients = new Array[Double](size)
    val hessians = new Array[Double](size)
    val maxNodeNum = Maths.pow(2, param.maxDepth + 1) - 1
    val nodePosStart = new Array[Int](maxNodeNum)
    val nodePosEnd = new Array[Int](maxNodeNum)
    val nodeToIns = new Array[Int](numData)
    val insPos = new Array[Int](numData)
    new CSRInstanceInfo(predictions, weights, gradients, hessians, nodePosStart, nodePosEnd, nodeToIns, insPos)
  }
}

class CSRInstanceInfo(override val predictions: Array[Float],
                      override val weights: Array[Float],
                      override val gradients: Array[Double],
                      override val hessians: Array[Double],
                      override val nodePosStart: Array[Int],
                      override val nodePosEnd: Array[Int],
                      override val nodeToIns: Array[Int],
                      val insPos: Array[Int])
  extends InstanceInfo(predictions, weights, gradients, hessians, nodePosStart, nodePosEnd, nodeToIns) {

  override def resetPosInfo(): Unit = {
    val num = weights.length
    nodePosStart(0) = 0
    nodePosEnd(0) = num - 1
    for (i <- 0 until num) {
      nodeToIns(i) = i
      insPos(i) = 0
    }
  }

  def getSplitResult2(nid: Int, fidInWorker: Int, splitEntry: SplitEntry, splits: Array[Float],
                      dataset: CSRDataset, threadPool: ExecutorService = null): RangeBitSet = {
    val res = new RangeBitSet(nodePosStart(nid), nodePosEnd(nid))

    val column = dataset.columns(fidInWorker)
    val indices = column.indices
    val bins = column.bins
    def split(start: Int, end: Int): Int = {
      var numSet = 0
      for (posId <- start until end) {
        val insId = nodeToIns(posId)
        val t = java.util.Arrays.binarySearch(indices, insId)
        val flowTo = if (t >= 0) {
          splitEntry.flowTo(splits(bins(t)))
        } else {
          splitEntry.defaultTo()
        }
        if (flowTo == 1) {
          res.set(posId)
          numSet += 1
        }
      }
      numSet
    }

    if (threadPool == null) {
      split(nodePosStart(nid), nodePosEnd(nid) + 1)
    } else {
      val numSet = ConcurrentUtil.rangeParallel(split, nodePosStart(nid), nodePosEnd(nid) + 1, threadPool)
        .map(_.get()).sum
      res.setNumSetTimes(numSet)
    }
    res
  }

  override def updatePos(nid: Int, splitResult: RangeBitSet): Unit = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val leftIns = new Array[Int](getNodeSize(nid) - splitResult.getNumSetTimes)
    val rightIns = new Array[Int](splitResult.getNumSetTimes)
    var l = 0
    var r = 0
    val leftNid = 2 * nid + 1
    val rightNid = 2 * nid + 2
    for (posId <- nodeStart to nodeEnd) {
      val insId = nodeToIns(posId)
      if (splitResult.get(posId)) {
        rightIns(r) = insId
        r += 1
        insPos(insId) = rightNid
      } else {
        leftIns(l) = insId
        l += 1
        insPos(insId) = leftNid
      }
    }
    require(r == splitResult.getNumSetTimes && l + r == getNodeSize(nid))
    System.arraycopy(leftIns, 0, nodeToIns, nodeStart, l)
    System.arraycopy(rightIns, 0, nodeToIns, nodeStart + l, r)
    nodePosStart(2 * nid + 1) = nodeStart
    nodePosEnd(2 * nid + 1) = nodeStart + l - 1
    nodePosStart(2 * nid + 2) = nodeStart + l
    nodePosEnd(2 * nid + 2) = nodeEnd
//    var left = nodeStart
//    var right = nodeEnd
//    while (left < right) {
//      while (left < right && !splitResult.get(left)) left += 1
//      while (left < right && splitResult.get(right)) right -= 1
//      if (left < right) {
//        val leftInsId = nodeToIns(left)
//        val rightInsId = nodeToIns(right)
//        nodeToIns(left) = rightInsId
//        nodeToIns(right) = leftInsId
//        insPos(leftInsId) = right
//        insPos(rightInsId) = left
//        left += 1
//        right -= 1
//      }
//    }
//    // find the cut position
//    val cutPos = if (left == right) {
//      if (splitResult.get(left)) left - 1
//      else left
//    } else {
//      right
//    }
//    nodePosStart(2 * nid + 1) = nodeStart
//    nodePosEnd(2 * nid + 1) = cutPos
//    nodePosStart(2 * nid + 2) = cutPos + 1
//    nodePosEnd(2 * nid + 2) = nodeEnd
  }

}
