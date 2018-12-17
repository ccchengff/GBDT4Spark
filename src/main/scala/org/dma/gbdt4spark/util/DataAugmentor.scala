package org.dma.gbdt4spark.util

import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random

object DataAugmentor {

  def parseLibsvm(line: String, dim: Int): (Int, Array[Int], Array[Double]) = {
    val splits = line.split("\\s+|,").map(_.trim)
    val y = splits(0).toInt

    val indices = new Array[Int](splits.length - 1)
    val values = new Array[Double](splits.length - 1)
    for (i <- 0 until splits.length - 1) {
      val kv = splits(i + 1).split(":")
      indices(i) = kv(0).toInt
      values(i) = kv(1).toDouble
    }

    (y, indices, values)
  }

  def instanceAug(ratio: Int) = (ins: (Int, Array[Int], Array[Double])) => {
    val values = ins._3
    for (i <- values.indices)
      values(i) += Random.nextGaussian() * 0.1
    ins
  }

  def featureAug(numFeature: Int, ratio: Int) = (ins: (Int, Array[Int], Array[Double])) => {
    val indices = ins._2
    val values = ins._3
    val nnz = indices.length
    val augIndices = new Array[Int](nnz * ratio)
    val augValues = new Array[Double](nnz * ratio)
    for (i <- 0 until ratio) {
      for (j <- 0 until nnz) {
        augIndices(i * nnz + j) = indices(j) + numFeature * i
        augValues(i * nnz + j) = values(j) + Random.nextGaussian() * 0.1
      }
    }
    (ins._1, augIndices, augValues)
  }

  def insToString = (ins: (Int, Array[Int], Array[Double])) => {
    val sb = new StringBuilder(s"${ins._1}")
    val indices = ins._2
    val values = ins._3
    for (i <- indices.indices) {
      sb.append(f" ${indices(i)}:${values(i)}%.3f")
    }
    sb.toString()
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    val sc = SparkContext.getOrCreate(conf)
    val input = conf.get("spark.ml.input.path")
    val output = conf.get("spark.ml.output.path")
    val numWorker = conf.get("spark.ml.worker.num").toInt
    val numFeature = conf.get("spark.ml.feature.num").toInt
    val numInsScaleRatio = conf.getInt("spark.ml.instance.ratio", 1)
    val numFeatureScaleRatio = conf.getInt("spark.ml.feature.ratio", 1)

    val data0 = sc.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => parseLibsvm(line, numFeature))
      .repartition(numWorker)
      .map(featureAug(numFeature, numFeatureScaleRatio))
      .cache()
    println(s"Data count: ${data.count()}")

    for (i <- 0 until numInsScaleRatio) {
      println(s"Copy $i saving to $output/$i")
      data.map(instanceAug(numInsScaleRatio))
        .map(insToString)
        .saveAsTextFile(s"$output/$i")
    }

    val data = sc.textFile("data/rcv1/t*")
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => parseLibsvm(line, 47237))
      .repartition(36)
      .cache()


    def myToString(label: Int, indices: Array[Int], values: Array[Double]): String = {
      val sb = new StringBuilder(s"$label")
      for (i <- indices.indices) {
        sb.append(f" ${indices(i)}:${values(i)}%.4f")
      }
      sb.toString()
    }


    import scala.collection.mutable
    import scala.util.Random
    val aug = data.map(ins => {
      val nnz = ins._2.length
      val indices = new mutable.ArrayBuilder.ofInt
      val values = new mutable.ArrayBuilder.ofDouble
      indices.sizeHint(nnz * 2)
      values.sizeHint(nnz * 2)
      for (i <- 0 until nnz) {
        indices += ins._2(i)
        values += ins._3(i)
      }
      for (i <- 0 until nnz) {
        if (Random.nextDouble() < 0.4) {
          indices += ins._2(i) + 47237
          if (ins._2(i) % 2 == 0)
            values += ins._3(i) + Random.nextGaussian()
          else
            values += ins._3(i)
        }
      }
      for (i <- (47237 * 2 + 1) to 10000) {
        if (Random.nextDouble() < 0.1) {
          indices += i
          values += Random.nextGaussian()
        }
      }
      myToString(ins._1, indices.result(), values.result())
    }).cache()

    for (i <- 0 until 70) {
      val output = s"ffc/data/gen/large/$i"
      println(s"Copy $i saving to $output")
      aug.saveAsTextFile(output)
    }

  }

  def tmp(): Unit = {
    import org.apache.spark.{SparkConf, SparkContext}
    import scala.collection.mutable
    import scala.util.Random

    def parseLibsvm(line: String, dim: Int): (Int, Array[Int], Array[Double]) = {
      val splits = line.split("\\s+|,").map(_.trim)
      val y = splits(0).toInt

      val indices = new Array[Int](splits.length - 1)
      val values = new Array[Double](splits.length - 1)
      for (i <- 0 until splits.length - 1) {
        val kv = splits(i + 1).split(":")
        indices(i) = kv(0).toInt
        values(i) = kv(1).toDouble
      }

      (y, indices, values)
    }

    def myToString(label: Int, indices: Array[Int], values: Array[Double]): String = {
      val sb = new StringBuilder(s"$label")
      for (i <- indices.indices) {
        sb.append(f" ${indices(i)}:${values(i)}%.4f")
      }
      sb.toString()
    }

    val conf = new SparkConf()
    val sc = SparkContext.getOrCreate(conf)
    val data = sc.textFile("data/rcv1/t*")
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => parseLibsvm(line, 47237))
      .repartition(10)
      .cache()
    val aug = data.map(ins => {
      val label = if (Random.nextGaussian() < 0.2) {
        if (Random.nextBoolean()) 1 else 0
      } else {
        ins._1
      }
      val nnz = ins._2.length
      val indices = new mutable.ArrayBuilder.ofInt
      val values = new mutable.ArrayBuilder.ofDouble
      indices.sizeHint(nnz * 2)
      values.sizeHint(nnz * 2)
      for (i <- 0 until nnz) {
        indices += ins._2(i)
        values += ins._3(i)
      }
      for (i <- 0 until nnz) {
        if (Random.nextDouble() < 0.4) {
          indices += ins._2(i) + 47237
          if (ins._2(i) % 2 == 0)
            values += ins._3(i) + Random.nextGaussian()
          else
            values += ins._3(i)
        }
      }
      for (i <- (47237 * 2 + 1) to 10000) {
        if (Random.nextDouble() < 0.1) {
          indices += i
          values += Random.nextGaussian()
        }
      }
      myToString(label, indices.result(), values.result())
    }).cache()
  }

}

