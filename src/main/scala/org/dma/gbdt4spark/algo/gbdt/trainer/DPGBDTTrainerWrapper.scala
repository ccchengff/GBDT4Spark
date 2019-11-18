package org.dma.gbdt4spark.algo.gbdt.trainer

import org.dma.gbdt4spark.algo.gbdt.histogram.GradPair
import org.dma.gbdt4spark.tree.split.SplitEntry

import scala.collection.mutable

private object DPGBDTTrainerWrapper {
  @transient private val trainers = mutable.Map[Int, DPGBDTTrainer]()

  private[trainer] def apply(workerId: Int, trainer: DPGBDTTrainer): DPGBDTTrainerWrapper = {
    trainers.synchronized {
      require(!trainers.contains(workerId), s"Id $workerId already exists")
      trainers += workerId -> trainer
      new DPGBDTTrainerWrapper(workerId)
    }
    //new DPGBDTTrainerWrapper(workerId)
  }
}

import DPGBDTTrainerWrapper._
private[trainer] class DPGBDTTrainerWrapper private(private[trainer] val workerId: Int) extends Serializable {

  private[trainer] def trainLabels = trainers(workerId).trainLabels

  private[trainer] def validLabels = trainers(workerId).validLabels

  private[trainer] def createNewTree() = trainers(workerId).createNewTree()

  //private[trainer] def buildHists() = trainers(workerId).buildHists()

  private[trainer] def buildHists(toBuild: Seq[Int], toSubtract: Seq[Boolean]) =
    trainers(workerId).buildHists(toBuild, toSubtract)

  private[trainer] def getNodeHists(nids: Seq[Int]) =
    trainers(workerId).getNodeHists(nids)

  private[trainer] def removeNodeHist(nid: Int) = trainers(workerId).removeNodeHist(nid)

  private[trainer] def splitNodes(splits: Map[Int, SplitEntry]) = trainers(workerId).splitNodes(splits)

  private[trainer] def setAsLeaf(nid: Int) = trainers(workerId).setAsLeaf(nid)

  //private[trainer] def finishTree() = trainers(workerId).finishTree()

  private[trainer] def finishTree(nodeGPs: Map[Int, GradPair]) = trainers(workerId).finishTree(nodeGPs)

  private[trainer] def evaluate() = trainers(workerId).evaluate()

  private[trainer] def finalizeModel() = trainers(workerId).finalizeModel()

}
