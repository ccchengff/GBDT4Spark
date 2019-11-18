package org.dma.gbdt4spark.algo.gbdt.trainer.csr

import org.dma.gbdt4spark.algo.gbdt.dataset.Dataset
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo

case class Column(indices: Array[Int], bins: Array[Int])


case class CSRDataset(columns: Array[Column]) {

  def numColumn: Int = columns.length

}


object CSRDataset {

  def apply(dataset: Dataset[Int, Int], featureInfo: FeatureInfo, featNNZ: Array[Int]): CSRDataset = {
    val numFeature = featureInfo.numFeature
    val colIndices = featNNZ.map(nnz => if (nnz > 0) new Array[Int](nnz) else null)
    val colValues = featNNZ.map(nnz => if (nnz > 0) new Array[Int](nnz) else null)
    val curIndexes = new Array[Int](numFeature)

    var offset = 0
    dataset.partitions.foreach(partition => {
      val numPartIns = partition.size
      val indices = partition.indices
      val bins = partition.values
      val indexEnds = partition.indexEnds
      var indexStart = 0
      for (i <- 0 until numPartIns) {
        val insId = offset + i
        val indexEnd = indexEnds(i)
        for (j <- indexStart until indexEnd) {
          val fid = indices(j)
          val binId = bins(j)
          colIndices(fid)(curIndexes(fid)) = insId
          colValues(fid)(curIndexes(fid)) = binId
          curIndexes(fid) += 1
        }
        indexStart = indexEnd
      }
      offset += numPartIns
    })

    val columns = (0 until numFeature).map(fid => Column(colIndices(fid), colValues(fid))).toArray
    CSRDataset(columns)
  }
}

