package org.dma.gbdt4spark.tree.param;

public class RegTParam extends TreeParam {
    public float learningRate;  // step size of one tree
    public float minSplitGain;  // minimum loss gain required for a split
}
