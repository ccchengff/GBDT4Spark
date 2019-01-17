package org.dma.gbdt4spark.algo.gbdt.trainer

import org.dma.gbdt4spark.algo.gbdt.tree.GBTSplit
import org.dma.gbdt4spark.util.RangeBitSet

import scala.collection.mutable

private object FPGBDTTrainerWrapper {
  @transient private val trainers = mutable.Map[Int, FPGBDTTrainer]()

  private[trainer] def apply(workerId: Int, trainer: FPGBDTTrainer): FPGBDTTrainerWrapper = {
    trainer.synchronized {
      require(!trainers.contains(workerId), s"Id $workerId already exists")
      trainers += workerId -> trainer
      new FPGBDTTrainerWrapper(workerId)
    }
  }
}

import FPGBDTTrainerWrapper._
private[trainer] class FPGBDTTrainerWrapper private(private[trainer] val workerId: Int) {

  private[trainer] def validLabels = trainers(workerId).validLabels

  private[trainer] def createNewTree() = trainers(workerId).createNewTree()

  private[trainer] def findSplits() = trainers(workerId).findSplits()

  private[trainer] def getSplitResults(splits: Seq[(Int, Int, Int, GBTSplit)]) =
    trainers(workerId).getSplitResults(splits)

  private[trainer] def splitNodes(splitResults: Seq[(Int, RangeBitSet)]) =
    trainers(workerId).splitNodes(splitResults)

  private[trainer] def setAsLeaf(nid: Int) = trainers(workerId).setAsLeaf(nid)

  private[trainer] def finishTree() = trainers(workerId).finishTree()

  private[trainer] def evaluate() = trainers(workerId).evaluate()

  private[trainer] def finalizeModel() = trainers(workerId).finalizeModel()

}
