package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.algo.gbdt.tree.GBTSplit;
import org.dma.gbdt4spark.tree.param.GBDTParam;
import org.dma.gbdt4spark.tree.split.SplitPoint;
import org.dma.gbdt4spark.tree.split.SplitSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SplitFinder {
    private static final Logger LOG = LoggerFactory.getLogger(SplitFinder.class);

    private final GBDTParam param;

    public SplitFinder(GBDTParam param) {
        this.param = param;
    }

    public GBTSplit findBestSplit(int[] fset, float[][] splits, Histogram[] histograms,
                                  GradPair sumGradPair, float nodeGain) {
        GBTSplit bestSplit = new GBTSplit();
        for (int i = 0; i < fset.length; i++) {
            int fid = fset[i];
            Histogram hist = histograms[i];
            GBTSplit gbtSplit = findBestSplitOfOneFeature(fid,
                    false, splits[fid], 0,
                    hist, sumGradPair, nodeGain);
            bestSplit.update(gbtSplit);
        }
        return bestSplit;
    }

    // TODO: use more schema on default bin
    public GBTSplit findBestSplitOfOneFeature(int fid, boolean isCategorical, float[] splits, int defaultBin,
                                              Histogram histogram, GradPair sumGradPair, float nodeGain) {
        if (isCategorical) {
            return findBestSplitSet(fid, splits, histogram, sumGradPair, nodeGain);
        } else {
            GBTSplit splitPoint = findBestSplitPoint(fid, splits, histogram, sumGradPair, nodeGain);
            GBTSplit splitSet = findBestSplitSet(fid, splits, histogram, sumGradPair, nodeGain);
            return splitPoint.needReplace(splitSet) ? splitSet : splitPoint;
        }
    }

    private GBTSplit findBestSplitPoint(int fid, float[] splits, Histogram histogram,
                                        GradPair sumGradPair, float nodeGain) {
        SplitPoint splitPoint = new SplitPoint();
        GradPair leftStat = param.numClass == 2 ? new BinaryGradPair()
                : new MultiGradPair(param.numClass, param.fullHessian);
        GradPair rightStat = sumGradPair.copy();
        GradPair bestLeftStat = null, bestRightStat = null;
        for (int i = 0; i < histogram.getNumSplit() - 1; i++) {
            leftStat.plusBy(histogram.get(i));
            rightStat.subtractBy(histogram.get(i));
            if (leftStat.satisfyWeight(param) && rightStat.satisfyWeight(param)) {
                float lossChg = leftStat.calcGain(param) + rightStat.calcGain(param)
                        - nodeGain - param.regLambda;
                if (splitPoint.needReplace(lossChg)) {
                    splitPoint.setFid(fid);
                    splitPoint.setFvalue(splits[i + 1]);
                    splitPoint.setGain(lossChg);
                    bestLeftStat = leftStat.copy();
                    bestRightStat = rightStat.copy();
                }
            }
        }
        return new GBTSplit(splitPoint, bestLeftStat, bestRightStat);
    }

    private GBTSplit findBestSplitSet(int fid, float[] splits, Histogram histogram,
                                      GradPair sumGradPair, float nodeGain) {
        SplitSet splitSet = new SplitSet();
        return new GBTSplit(splitSet, null, null);
    }
}
