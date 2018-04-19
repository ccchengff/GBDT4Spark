package org.dma.gbdt4spark.tree.split;

import org.apache.spark.ml.linalg.Vector;

import java.util.Arrays;

public class SplitSet extends SplitEntry {
    private float[] edges;

    @Override
    public int flowTo(float x) {
        return 0;
    }

    @Override
    public int flowTo(Vector x) {
        return 0;
    }

    @Override
    public int defaultTo() {
        return 0;
    }

    @Override
    public SplitType splitType() {
        return SplitType.SPLIT_SET;
    }

    @Override
    public String toString() {
        return String.format("%s fid[%d] edges%s gain[%f]",
                this.splitType(), fid, Arrays.toString(edges), gain);
    }
}
