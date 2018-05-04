package org.dma.gbdt4spark.algo.gbdt.metadata

import java.util

import org.dma.gbdt4spark.algo.gbdt.histogram.{BinaryGradPair, GradPair, MultiGradPair}
import org.dma.gbdt4spark.data.FeatureRow
import org.dma.gbdt4spark.objective.loss.{BinaryLoss, Loss, MultiLoss}
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util.{Maths, RangeBitSet}

object DataInfo {
  def apply(param: GBDTParam, numData: Int): DataInfo = {
    val size = if (param.numClass == 2) numData else param.numClass * numData
    val predictions = new Array[Float](size)
    val weights = Array.fill[Float](numData)(1.0f)
    val gradParis = new Array[GradPair](numData)
    val maxNodeNum = Maths.pow(2, param.maxDepth + 1) - 1
    val nodePosStart = new Array[Int](maxNodeNum)
    val nodePosEnd = new Array[Int](maxNodeNum)
    val nodeToIns = new Array[Int](numData)
    val insPos = new Array[Int](numData)
    new DataInfo(predictions, weights, gradParis, nodePosStart, nodePosEnd, nodeToIns, insPos)
  }

}

case class DataInfo(predictions: Array[Float], weights: Array[Float], gradPairs: Array[GradPair],
                    nodePosStart: Array[Int], nodePosEnd: Array[Int], nodeToIns: Array[Int], insPos: Array[Int]) {

  def resetPosInfo(): Unit = {
    val num = weights.length
    nodePosStart(0) = 0
    nodePosEnd(0) = num - 1
    for (i <- 0 until num) {
      nodeToIns(i) = i
      insPos(i) = i
    }
  }

  def calcGradPairs(nid: Int, labels: Array[Float], loss: Loss, param: GBDTParam): GradPair = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val numClass = param.numClass
    if (numClass == 2) {
      // binary classification
      val binaryLoss = loss.asInstanceOf[BinaryLoss]
      val sumGradPair = new BinaryGradPair()
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        val grad = binaryLoss.firOrderGrad(predictions(insId), labels(insId))
        val hess = binaryLoss.secOrderGrad(predictions(insId), labels(insId), grad)
        gradPairs(insId) = new BinaryGradPair(grad, hess)
        sumGradPair.plusBy(gradPairs(insId))
      }
      sumGradPair
    } else if (!param.fullHessian) {
      // multi-label classification, assume hessian matrix is diagonal
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val sumGradPair = new MultiGradPair(numClass, false)
      val preds = new Array[Float](numClass)
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradDiag(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
        sumGradPair.plusBy(gradPairs(insId))
      }
      sumGradPair
    } else {
      // multi-label classification, represent hessian matrix as lower triangular matrix
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val sumGradPair = new MultiGradPair(numClass, true)
      val preds = new Array[Float](numClass)
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradFull(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
        sumGradPair.plusBy(gradPairs(insId))
      }
      sumGradPair
    }
  }

  def sumGradPair(nid: Int): GradPair = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    require(nodeStart <= nodeEnd)
    val res = gradPairs(nodeToIns(nodeStart)).copy()
    for (i <- nodeStart + 1 to nodeEnd)
      res.plusBy(gradPairs(nodeToIns(i)))
    res
  }

  def getSplitResult(nid: Int, splitEntry: SplitEntry,
                     featureRow: FeatureRow, splits: Array[Float]): RangeBitSet = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val res = new RangeBitSet(nodeStart, nodeEnd)
    val defaultTo = splitEntry.defaultTo()
    for (posId <- nodeStart to nodeEnd) {
      val insId = nodeToIns(posId)
      val index = util.Arrays.binarySearch(featureRow.indices, insId)
      val flowTo = if (index >= 0) {
        val value = splits(featureRow.bins(index))
        splitEntry.flowTo(value)
      } else {
        defaultTo
      }
      if (flowTo == 1)
        res.set(posId)
    }
    res
  }

  def updatePos(nid: Int, splitResult: RangeBitSet): (Int, Int) = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    var left = nodeStart
    var right = nodeEnd
    while (left < right) {
      while (left < right && !splitResult.get(left)) left += 1
      while (left < right && splitResult.get(right)) right -= 1
      if (left < right) {
        val leftInsId = nodeToIns(left)
        val rightInsId = nodeToIns(right)
        nodeToIns(left) = rightInsId
        nodeToIns(right) = leftInsId
        insPos(leftInsId) = right
        insPos(rightInsId) = left
        left += 1
        right -= 1
      }
    }
    // find the cut position
    val cutPos = if (left == right) {
      if (splitResult.get(left)) left - 1
      else left
    } else {
      right
    }
    nodePosStart(2 * nid + 1) = nodeStart
    nodePosEnd(2 * nid + 1) = cutPos
    val leftChildSize = cutPos - nodeStart + 1
    nodePosStart(2 * nid + 2) = cutPos + 1
    nodePosEnd(2 * nid + 2) = nodeEnd
    val rightChildSize = nodeEnd - cutPos
    (leftChildSize, rightChildSize)
  }

  def updatePreds(nid: Int, update: Float, learningRate: Float): Unit = {
    val update_ = update * learningRate
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    for (i <- nodeStart to nodeEnd) {
      val insId = nodeToIns(i)
      predictions(insId) += update_
    }
  }

  def updatePreds(nid: Int, update: Array[Float], learningRate: Float): Unit = {
    val numClass = update.length
    val update_ = update.map(_ * learningRate)
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    for (i <- nodeStart to nodeEnd) {
      val insId = nodeToIns(i)
      val offset = insId * numClass
      for (k <- 0 until numClass)
        predictions(offset + k) += update_(k)
    }
  }

  def getNodePosStart(nid: Int) = nodePosStart(nid)

  def getNodePosEnd(nid: Int) = nodePosEnd(nid)
}
