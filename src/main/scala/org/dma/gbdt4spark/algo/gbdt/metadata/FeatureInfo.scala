package org.dma.gbdt4spark.algo.gbdt.metadata

import org.dma.gbdt4spark.util.Maths

object FeatureInfo {
  def apply(featTypes: Array[Boolean], splits: Array[Array[Float]]): FeatureInfo = {
    require(featTypes.length == splits.length)
    val numFeature = featTypes.length
    val numBin = new Array[Int](numFeature)
    val defaultBins = new Array[Int](numFeature)
    for (i <- 0 until numFeature) {
      if (featTypes(i)) {
        numBin(i) = splits(i).length + 1
        defaultBins(i) = splits(i).length
      } else {
        numBin(i) = splits(i).length
        defaultBins(i) = Maths.indexOf(splits(i), 0.0f)  // TODO: default bin for continuous feature
      }
    }

    new FeatureInfo(featTypes, numBin, splits, defaultBins)
  }
}

case class FeatureInfo(featTypes: Array[Boolean], numBin: Array[Int],
                       splits: Array[Array[Float]], defaultBins: Array[Int]) {
  def isCategorical(fid: Int) = featTypes(fid)

  def getNumBin(fid: Int) = numBin(fid)

  def getSplits(fid: Int) = splits(fid)

  def getDefaultBin(fid: Int) = defaultBins(fid)
}
