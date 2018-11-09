package org.dma.gbdt4spark.algo.gbdt.metadata

import java.util

import it.unimi.dsi.fastutil.ints.IntArrayList
import org.dma.gbdt4spark.algo.gbdt.histogram.{BinaryGradPair, GradPair, MultiGradPair}
import org.dma.gbdt4spark.data.{FeatureRow, InstanceRow}
import org.dma.gbdt4spark.objective.loss.{BinaryLoss, Loss, MultiLoss}
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.tree.split.SplitEntry
import org.dma.gbdt4spark.util._

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
    if (nodeToInsCopy == null)
      nodeToInsCopy = new Array[Int](numData)
    //val insToFeat = new Array[IntOpenHashSet](numData)
    //for (i <- 0 until numData) insToFeat(i) = new IntOpenHashSet()
    //new DataInfo(predictions, weights, gradParis, nodePosStart, nodePosEnd, nodeToIns, insPos, insToFeat)
    new DataInfo(predictions, weights, gradParis, nodePosStart, nodePosEnd, nodeToIns, insPos)
  }

  private var nodeToInsCopy: Array[Int] = _

}

case class DataInfo(predictions: Array[Float], weights: Array[Float], gradPairs: Array[GradPair],
                    nodePosStart: Array[Int], nodePosEnd: Array[Int], nodeToIns: Array[Int], insPos: Array[Int]) {

  //def inputNNZ(insId: Int, fid: Int): Unit = insToFeat(insId).add(fid)

  def resetPosInfo(): Unit = {
    val num = weights.length
    nodePosStart(0) = 0
    nodePosEnd(0) = num - 1
    for (i <- 0 until num) {
      nodeToIns(i) = i
      insPos(i) = i
    }
  }

  def calcGradPairs(nid: Int, labels: Array[Float], loss: Loss, param: GBDTParam): Unit = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val numClass = param.numClass
    if (numClass == 2) {
      // binary classification
      val binaryLoss = loss.asInstanceOf[BinaryLoss]
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        val grad = binaryLoss.firOrderGrad(predictions(insId), labels(insId))
        val hess = binaryLoss.secOrderGrad(predictions(insId), labels(insId), grad)
        gradPairs(insId) = new BinaryGradPair(grad, hess)
      }
    } else if (!param.fullHessian) {
      // multi-label classification, assume hessian matrix is diagonal
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val preds = new Array[Float](numClass)
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradDiag(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
      }
    } else {
      // multi-label classification, represent hessian matrix as lower triangular matrix
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val preds = new Array[Float](numClass)
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradFull(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
      }
    }
  }

  def parallelCalcGradPairs(nid: Int, labels: Array[Float], loss: Loss,
                            param: GBDTParam, partId: Int, numPart: Int): Unit = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val avg = (nodeEnd - nodeStart + 1) / numPart
    val from = nodeStart + avg * partId
    val to = if (partId + 1 == numPart) nodeEnd else from + avg
    val numClass = param.numClass
    if (numClass == 2) {
      // binary classification
      val binaryLoss = loss.asInstanceOf[BinaryLoss]
      for (posId <- from to to) {
        val insId = nodeToIns(posId)
        val grad = binaryLoss.firOrderGrad(predictions(insId), labels(insId))
        val hess = binaryLoss.secOrderGrad(predictions(insId), labels(insId), grad)
        gradPairs(insId) = new BinaryGradPair(grad, hess)
      }
    } else if (!param.fullHessian) {
      // multi-label classification, assume hessian matrix is diagonal
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val preds = new Array[Float](numClass)
      for (posId <- from to to) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradDiag(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
      }
    } else {
      // multi-label classification, represent hessian matrix as lower triangular matrix
      val multiLoss = loss.asInstanceOf[MultiLoss]
      val preds = new Array[Float](numClass)
      for (posId <- from to to) {
        val insId = nodeToIns(posId)
        Array.copy(predictions, insId * numClass, preds, 0, numClass)
        val grad = multiLoss.firOrderGrad(preds, labels(insId))
        val hess = multiLoss.secOrderGradFull(preds, labels(insId), grad)
        gradPairs(insId) = new MultiGradPair(grad, hess)
      }
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
                     splits: Array[Float], instances: Array[InstanceRow]): RangeBitSet = {
    val res = new RangeBitSet(nodePosStart(nid), nodePosEnd(nid))
    val fid = splitEntry.getFid
    for (posId <- nodePosStart(nid) to nodePosEnd(nid)) {
      val ins = instances(nodeToIns(posId))
      val t = java.util.Arrays.binarySearch(ins.indices, fid)
      val flowTo = if (t >= 0) {
        splitEntry.flowTo(splits(ins.bins(t)))
      } else {
        splitEntry.defaultTo()
      }
      if (flowTo == 1)
        res.set(posId)
    }
    res
  }

  def getSplitResult(nid: Int, splitEntry: SplitEntry,
                     featureRow: FeatureRow, splits: Array[Float]): RangeBitSet = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val res = new RangeBitSet(nodeStart, nodeEnd)
    //val writer = new BufferedBitSetWriter(nodeEnd - nodeStart + 1)
    //val flowToLeft = new IntArrayList()
    //val flowToRight = new IntArrayList()
    val defaultTo = splitEntry.defaultTo()
    val indices = featureRow.indices
    val bins = featureRow.bins
    val nnz = indices.length
    if ((nodeEnd - nodeStart + 1) >= nnz / 100) {
      var posId = nodeStart
      var indId = 0
      while (posId <= nodeEnd && indId < nnz) {
        val insId = nodeToIns(posId)
        if (insId == indices(indId)) {
          val value = splits(bins(indId))
          //writer.write(splitEntry.flowTo(value) == 1)
          //if (splitEntry.flowTo(value) == 1) flowToRight.add(insId)
          //else flowToLeft.add(insId)
          if (splitEntry.flowTo(value) == 1)
            res.set(posId)
          posId += 1
          indId += 1
        } else if (insId > indices(indId)) {
          indId += 1
        } else {
          //writer.write(defaultTo == 1)
          //if (defaultTo == 0) flowToLeft.add(insId)
          //else flowToRight.add(insId)
          if (defaultTo == 1)
            res.set(posId)
          posId += 1
        }
      }
      if (defaultTo == 1) {
        while (posId <= nodeEnd) {
          //writer.write(true)
          //flowToRight.add(nodeToIns(posId))
          res.set(posId)
          posId += 1
        }
      } //else {
        //while (posId <= nodeEnd) {
        //  //writer.write(false)
        //  //flowToLeft.add(nodeToIns(posId))
        //  //res.set(posId) //!!!!!
        //  posId += 1
        //}
      //}
    } else {
      for (posId <- nodeStart to nodeEnd) {
        val insId = nodeToIns(posId)
        val index = util.Arrays.binarySearch(featureRow.indices, insId)
        val flowTo = if (index >= 0) {
          val value = splits(featureRow.bins(index))
          splitEntry.flowTo(value)
        } else {
          defaultTo
        }
        //writer.write(flowTo == 1)
        if (flowTo == 1)
          res.set(posId)
        //if (flowTo == 0) flowToLeft.add(insId)
        //else flowToRight.add(insId)
      }
    }
    /*for (posId <- nodeStart to nodeEnd) {
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
      //writer.write(flowTo == 1)
    }*/
    res
    //(flowToLeft.toIntArray(null), flowToRight.toIntArray(null))
    //writer.complete()
    //writer.getBytes
  }

  //def updatePos(nid: Int, flowTos: (Array[Int], Array[Int])): Unit = {
  def updatePos(nid: Int, splitResult: RangeBitSet): Unit = {
    val nodeStart = nodePosStart(nid)
    val nodeEnd = nodePosEnd(nid)
    val copy = nodeToIns.slice(nodeStart, nodeEnd + 1)
    var posL = nodeStart
    var posR = nodeEnd - splitResult.getNumSetTimes + 1
    for (posId <- nodeStart to nodeEnd) {
      val insId = copy(posId - nodeStart)
      if (splitResult.get(posId)) {
        nodeToIns(posR) = insId
        insPos(insId) = posR
        posR += 1
      } else {
        nodeToIns(posL) = insId
        insPos(insId) = posL
        posL += 1
      }
    }
    require(posL + splitResult.getNumSetTimes == posR && posR == nodeEnd + 1)
    val cutPos = posL - 1
    /*val flowToLeft = flowTos._1
    val flowToRight = flowTos._2
    for (i <- flowToLeft.indices) {
      nodeToIns(nodeStart + i) = flowToLeft(i)
      insPos(flowToLeft(i)) = nodeStart + i
    }
    for (i <- flowToRight.indices) {
      nodeToIns(nodeStart + flowToLeft.length + i) = flowToRight(i)
      insPos(flowToRight(i)) = nodeStart + flowToLeft.length + i
    }
    val cutPos = nodeStart + flowToLeft.length - 1*/
    /*var left = nodeStart
    var right = nodeEnd
    //val reader = new BufferedBitSetReader(bytes, nodeEnd - nodeStart + 1)
    while (left < right) {
      while (left < right && !splitResult.get(left)) left += 1
      while (left < right && splitResult.get(right)) right -= 1
      //while (left < right && !reader.readHead()) left += 1
      //while (left < right && reader.readTail()) right -= 1
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
      //if (reader.read(left)) left - 1
      else left
    } else {
      right
    }*/
    nodePosStart(2 * nid + 1) = nodeStart
    nodePosEnd(2 * nid + 1) = cutPos
    nodePosStart(2 * nid + 2) = cutPos + 1
    nodePosEnd(2 * nid + 2) = nodeEnd
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

  def getNodeSize(nid: Int): Int = nodePosEnd(nid) - nodePosStart(nid) + 1

}
