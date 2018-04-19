package org.dma.gbdt4spark.tree.param;

import java.io.Serializable;

public abstract class TreeParam implements Serializable {
    public int numFeature;  // number of features
    public int maxDepth;  // maximum depth
    public int maxNodeNum;  // maximum node num
    public int numSplit;  // number of candidate splits
    public int numWorker;  // number of workers
    public float insSampleRatio;  // subsample ratio for instances
    public float featSampleRatio;  // subsample ratio for features
}
