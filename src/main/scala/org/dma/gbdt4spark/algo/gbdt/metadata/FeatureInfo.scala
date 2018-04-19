package org.dma.gbdt4spark.algo.gbdt.metadata

import org.dma.gbdt4spark.util.Maths

object FeatureInfo {
  def apply(featTypes: Array[Boolean], splits: Array[Array[Float]]): FeatureInfo = {
    val numSplits = splits.map(_.length)
    val defaultBins = (featTypes, splits).zipped.map((t, s) => {
      if (t) s.length - 1
      else Maths.indexOf(s, 0.0f)  // TODO: default bin for continuous feature
    })
    new FeatureInfo(featTypes, numSplits, splits, defaultBins)
  }
}

case class FeatureInfo(featTypes: Array[Boolean], numSplits: Array[Int],
                       splits: Array[Array[Float]], defaultBins: Array[Int]) {
  def isCategorical(fid: Int) = featTypes(fid)

  def getNumSplit(fid: Int) = numSplits(fid)

  def getSplits(fid: Int) = splits(fid)

  def getDefaultBin(fid: Int) = defaultBins(fid)
}
