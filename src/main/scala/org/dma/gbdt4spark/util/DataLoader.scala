package org.dma.gbdt4spark.util

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.dma.gbdt4spark.data.Instance

object DataLoader {
  def loadLibsvm(input: String, dim: Int)(implicit sc: SparkContext): RDD[Instance] = {
    sc.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => parseLibsvm(line, dim))
  }

  def parseLibsvm(line: String, dim: Int): Instance = {
    val splits = line.split("\\s+|,").map(_.trim)
    val y = splits(0).toDouble

    val indices = new Array[Int](splits.length - 1)
    val values = new Array[Double](splits.length - 1)
    for (i <- 0 until splits.length - 1) {
      val kv = splits(i + 1).split(":")
      indices(i) = kv(0).toInt
      values(i) = kv(1).toDouble
    }

    Instance(y, Vectors.sparse(dim, indices, values))
  }

}
