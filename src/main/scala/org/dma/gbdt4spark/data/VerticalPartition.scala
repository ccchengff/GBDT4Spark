package org.dma.gbdt4spark.data

import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo
import org.dma.gbdt4spark.sketch.HeapQuantileSketch
import org.dma.gbdt4spark.util.Maths

object VerticalPartition {

  def getCandidateSplits(partitions: Seq[VerticalPartition],
                         numFeature: Int, numSplit: Int): Seq[(Int, Array[Float])] = {
    val featNNZ = new Array[Int](numFeature)
    partitions.foreach(_.indices.foreach(fid => featNNZ(fid) += 1))
    val sketches = featNNZ.map(nnz =>
      if (nnz > 0) new HeapQuantileSketch(nnz.toLong) else null)
    partitions.foreach(partition => {
      val partIndices = partition.indices
      val partValues = partition.values
      for (i <- partIndices.indices)
        sketches(partIndices(i)).update(partValues(i))
    })

    sketches.view.zipWithIndex.flatMap {
      case (sketch, fid) =>
        if (sketch != null && sketch.getN > 0)
          Iterator((fid, Maths.unique(sketch.getQuantiles(numSplit))))
        else
          Iterator.empty
    }
  }

  def discretize(partitions: Seq[VerticalPartition],
                 featureInfo: FeatureInfo): (Array[Float], Array[InstanceRow]) = {
    val numInstance = partitions.map(_.labels.length).sum
    val labels = new Array[Float](numInstance)
    val instances = new Array[InstanceRow](numInstance)
    var offset = 0
    partitions.sortBy(_.originPartId)
      .foreach(partition => {
        val partSize = partition.labels.length
        Array.copy(partition.labels, 0, labels, offset, partSize)
        val partIndexEnd = partition.indexEnd
        val partIndices = partition.indices
        val partValues = partition.values
        var partOffset = 0
        for (i <- 0 until partSize) {
          val insNNZ = partIndexEnd(i) - partOffset
          val indices = partIndices.slice(partOffset, partIndexEnd(i))
          val bins = new Array[Int](insNNZ)
          for (j <- 0 until insNNZ) {
            bins(j) = Maths.indexOf(featureInfo.getSplits(indices(j)),
              partValues(partOffset + j))
          }
          partOffset += insNNZ
          instances(offset + i) = InstanceRow(indices, bins)
        }
        offset += partSize
      })
    (labels, instances)
  }
}

case class VerticalPartition(originPartId: Int, labels: Array[Float], indexEnd: Array[Int],
                             indices: Array[Int], values: Array[Float])
