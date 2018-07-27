package org.dma.gbdt4spark.data


case class InstanceRow(indices: Array[Int], bins: Array[Int]) {

  def size: Int = if (indices == null) 0 else indices.length
}