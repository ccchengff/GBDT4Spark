package org.dma.gbdt4spark.algo.gbdt.tree;

import org.dma.gbdt4spark.algo.gbdt.histogram.GradPair;
import org.dma.gbdt4spark.tree.basic.TNodeStat;
import org.dma.gbdt4spark.tree.param.GBDTParam;

public class GBTNodeStat extends TNodeStat {
    /*private GradPair sumGradPair;

    public GBTNodeStat(GradPair sumGradPair) {
        this.sumGradPair = sumGradPair;
    }

    public float calcWeight(GBDTParam param) {
        nodeWeight = param.calcWeight(sumGrad, sumHess);
        return nodeWeight;
    }

    public float calcGain(GBDTParam param) {
        gain = param.calcGain(sumGrad, sumHess);
        return gain;
    }

    public float getSumGrad() {
        return sumGrad;
    }

    public float getSumHess() {
        return sumHess;
    }

    public void setSumGrad(float sumGrad) {
        this.sumGrad = sumGrad;
    }

    public void setSumHess(float sumHess) {
        this.sumHess = sumHess;
    }*/
}
