package org.dma.gbdt4spark.util

import org.apache.spark.util.LongAccumulator
import org.apache.spark.{SparkConf, SparkContext}
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo
import org.dma.gbdt4spark.common.Global.Conf._
import org.dma.gbdt4spark.sketch.HeapQuantileSketch
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

object DatasetAnalysis {

  private val LOG = LoggerFactory.getLogger(DatasetAnalysis.getClass)

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GBDT")
    implicit val sc = SparkContext.getOrCreate(conf)

    if (args(0) == "change_label")
      change_label(conf)
    else if (args(0) == "coalesce_label")
      coalesce_label(conf)
    else if (args(0) == "analysis")
      analysis(conf)
    else if (args(0) == "shuffle_feature")
      shuffle_feature(conf)
  }

  def analysis(conf: SparkConf)(implicit sc: SparkContext): Unit = {
    val input = conf.get(ML_TRAIN_DATA_PATH)
    val dim = conf.get(ML_NUM_FEATURE).toInt
    val numWorker = conf.get(ML_NUM_WORKER).toInt
    val numSplit = conf.getInt(ML_GBDT_SPLIT_NUM, DEFAULT_ML_GBDT_SPLIT_NUM)

    val loadStart = System.currentTimeMillis()

    val dataset = sc.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => DataLoader.parseLibsvm(line, dim))
      .persist()
    val numData = dataset.count()
    val minNNZ  = dataset.map(_.feature.numActives).reduce(_ min _)
    val maxNNZ  = dataset.map(_.feature.numActives).reduce(_ max _)
    val numKVPair = dataset.map(_.feature.numActives).reduce(_ + _)
    val minFeat = dataset.map(instance => {
      var res = Int.MaxValue
      instance.feature.foreachActive((k, v) => res = res min k)
      res
    }).reduce(_ min _)
    val maxFeat = dataset.map(instance => {
      var res = Int.MinValue
      instance.feature.foreachActive((k, v) => res = res max k)
      res
    }).reduce(_ max _)

    LOG.info(s"Load data cost ${System.currentTimeMillis() - loadStart} ms, " +
      s"$numData instances, $numKVPair kv pairs, min nnz = $minNNZ, max nnz = $maxNNZ, " +
      s"min feat = $minFeat, max feat = $maxFeat")

    val featNNZAcc = (0 until dim).map(fid => fid -> sc.longAccumulator).toMap
    dataset.foreachPartition(iter => {
      val cnt = new Array[Long](dim)
      while (iter.hasNext) {
        iter.next().feature.foreachActive((k, v) => cnt(k) += 1)
      }
      for (i <- 0 until dim) {
        featNNZAcc.get(i) match {
          case Some(t) => t.add(cnt(i))
          case None =>  // should not happen
        }
      }
    })
    val featNNZ = featNNZAcc.toSeq.sortWith(_._2.value < _._2.value)
    val featNNZSketch = new HeapQuantileSketch()
    featNNZ.foreach(p => featNNZSketch.update(p._2.value.toFloat))
    LOG.info(s"Min feat nnz: [fid = ${featNNZ.head._1}, ${featNNZ.head._2.value}")
    LOG.info(s"Min feat nnz: [fid = ${featNNZ.last._1}, ${featNNZ.last._2.value}")
    LOG.info(s"Feat nnz quantiles: [${featNNZSketch.getQuantiles(100)}]")

    val labels = dataset.mapPartitions(iter => {
      val map = collection.mutable.Map[Double, Long]()
      while (iter.hasNext) {
        val label = iter.next().label
        map += label -> (map.getOrElse(label, 0L) + 1L)
      }
      Seq(map).iterator
    }).reduce((m1, m2) => {
      m2.foreach(p => {
        m1 += p._1 -> (m1.getOrElse(p._1, 0L) + p._2)
      })
      m1
    }).toSeq.sortWith(_._1 < _._1)
    LOG.info(s"Labels: [${labels.mkString(", ")}]")

    val evenPartitioner = new EvenPartitioner(dim, numWorker)
    val featureEdges = evenPartitioner.partitionEdges()
    val bcFeatureEdges = sc.broadcast(featureEdges)

    val getSplitStart = System.currentTimeMillis()
    val splits = new Array[Array[Float]](dim)
    dataset.mapPartitions(iter => {
      val sketches = new Array[(Int, HeapQuantileSketch)](dim)
      for (i <- 0 until dim) {
        sketches(i) = (i, new HeapQuantileSketch())
      }
      while (iter.hasNext) {
        iter.next().feature.foreachActive((k, v) => sketches(k)._2.update(v.toFloat))
      }
      sketches.filter(!_._2.isEmpty).iterator
    }).repartitionAndSortWithinPartitions(evenPartitioner)
      .mapPartitionsWithIndex((partId, iter) => {
        //val featLo = bcFeatureEdges.value(partId)
        //val featHi = bcFeatureEdges.value(partId + 1)
        val splits = collection.mutable.ArrayBuffer[(Int, Array[Float])]()
        //val splits = new Array[(Int, Array[Float])](featHi - featLo)
        var curFid = -1
        var curSketch: HeapQuantileSketch = null
        while (iter.hasNext) {
          val (fid, sketch) = iter.next()
          if (fid != curFid) {
            if (curFid != -1) {
              splits += ((curFid, Maths.unique(curSketch.getQuantiles(numSplit))))
            }
            curSketch = sketch
            curFid = fid
          } else {
            curSketch.merge(sketch)
          }
        }
        splits.iterator
      }, preservesPartitioning = true)
      .collect()
      .foreach(s => splits(s._1) = s._2)

    LOG.info(s"Get candidate splits cost ${System.currentTimeMillis() - getSplitStart} ms")
    val featureInfo = FeatureInfo(dim, splits)

    //val uniqNumSplits = Maths.unique(featureInfo.splits.filter(_ != null).
    //  map(_.length.toFloat).sortWith(_ < _))
    //LOG.info(s"Num splits: [${uniqNumSplits.mkString(", ")}]")
    val numSplitCnt = new Array[Int](numSplit + 1)
    featureInfo.splits.map(s => if (s == null) 0 else s.length).foreach(numSplitCnt(_) += 1)
    LOG.info(s"Num split count: [${numSplitCnt.mkString(", ")}]")

    val sketch = new HeapQuantileSketch()
    featureInfo.splits.foreach(s => if (s != null) sketch.update(s.length))
    LOG.info(s"10 quantiles of number of candidate splits: [${sketch.getQuantiles(10).mkString(", ")}]")
    LOG.info(s"25 quantiles of number of candidate splits: [${sketch.getQuantiles(25).mkString(", ")}]")
    LOG.info(s"50 quantiles of number of candidate splits: [${sketch.getQuantiles(50).mkString(", ")}]")
    LOG.info(s"100 quantiles of number of candidate splits: [${sketch.getQuantiles(100).mkString(", ")}]")
  }

  def change_label(conf: SparkConf)(implicit sc: SparkContext): Unit = {
    val input = conf.get(ML_TRAIN_DATA_PATH)
    val output = conf.get("spark.ml.output.path")

    sc.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => {
        var index = line.indexOf(" ")
        if (index == -1)
          index = line.length - 1
        val label = line.substring(0, index).toInt - 1
        label.toString + " " + line.substring(index)
      }).saveAsTextFile(output)
  }

  def coalesce_label(conf: SparkConf)(implicit sc: SparkContext): Unit = {
    val input = conf.get(ML_TRAIN_DATA_PATH)
    val output = conf.get("spark.ml.output.path")
    val numClass = conf.get(ML_NUM_CLASS).toInt
    val coalescedNumClass = conf.get("spark.ml.coalesced.class.num").toInt

    val avg = if (numClass % coalescedNumClass > numClass / 2) {
      Math.ceil(1.0 * numClass / coalescedNumClass).toInt
    } else {
      Math.floor(1.0 * numClass / coalescedNumClass).toInt
    }

    sc.textFile(input)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => {
        var index = line.indexOf(" ")
        if (index == -1)
          index = line.length
        val oriLabel = line.substring(0, index).toInt
        val label = (oriLabel / avg) min (coalescedNumClass - 1)
        label.toString + " " + line.substring(index)
      }).saveAsTextFile(output)
  }

  def shuffle_feature(conf: SparkConf)(implicit sc: SparkContext): Unit = {
    val input = conf.get(ML_TRAIN_DATA_PATH)
    val output = conf.get("spark.ml.output.path")
    val numFeature = conf.get(ML_NUM_FEATURE).toInt
    val shuffle = (0 until numFeature).toArray
    Maths.shuffle(shuffle)
    val bcShuffle = sc.broadcast(shuffle)

    DataLoader.loadLibsvmDP(input, numFeature)
      .map(instance => {
        val kvs = collection.mutable.ArrayBuffer[(Int, Double)]()
        instance.feature.foreachActive((k, v) => kvs += ((bcShuffle.value(k), v)))
        val sb = new StringBuilder
        sb.append(instance.label.toInt)
        kvs.sortWith(_._1 < _._1).foreach(kv => sb.append(s" ${kv._1}:${kv._2}"))
        sb.toString()
      }).saveAsTextFile(output)
  }
}
