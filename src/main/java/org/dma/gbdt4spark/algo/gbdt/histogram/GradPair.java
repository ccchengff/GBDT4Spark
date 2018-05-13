package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.tree.param.GBDTParam;

public interface GradPair {
    void plusBy(GradPair gradPair);

    void subtractBy(GradPair gradPair);

    GradPair plus(GradPair gradPair);

    GradPair subtract(GradPair gradPair);

    void timesBy(double x);

    float calcGain(GBDTParam param);

    boolean satisfyWeight(GBDTParam param);

    GradPair copy();

}
