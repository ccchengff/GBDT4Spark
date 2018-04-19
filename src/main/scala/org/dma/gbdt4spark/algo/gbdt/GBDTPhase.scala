package org.dma.gbdt4spark.algo.gbdt

object GBDTPhase extends Enumeration {
  type GBDTPhase = Value
  val NEW_TREE, CHECK_STATUS, BUILD_HIST, FIND_SPLIT, SPLIT_NODE, FINISH_TREE, FINISHED = Value
}
