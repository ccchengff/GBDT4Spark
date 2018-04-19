package org.dma.gbdt4spark.algo.gbdt.tree;

import org.dma.gbdt4spark.algo.gbdt.histogram.GradPair;
import org.dma.gbdt4spark.tree.split.SplitEntry;

import java.io.Serializable;

public class GBTSplit implements Serializable {
    private SplitEntry splitEntry;
    private GradPair leftGradPair;  // grad pair of left child
    private GradPair rightGradPair; // grad pair of right child

    public GBTSplit() {
        this(null, null, null);
    }

    public GBTSplit(SplitEntry splitEntry, GradPair leftGradPair, GradPair rightGradPair) {
        super();
        this.splitEntry = splitEntry;
        this.leftGradPair = leftGradPair;
        this.rightGradPair = rightGradPair;
    }

    public boolean needReplace(GBTSplit split) {
        if (this.splitEntry != null)
            return split.splitEntry != null && this.splitEntry.needReplace(split.splitEntry);
        else
            return split.splitEntry != null;
    }

    public void update(GBTSplit split) {
        if (this.needReplace(split)) {
            this.splitEntry = split.splitEntry;
            this.leftGradPair = split.leftGradPair;
            this.rightGradPair = split.rightGradPair;
        }
    }

    public SplitEntry getSplitEntry() {
        return splitEntry;
    }

    public void setSplitEntry(SplitEntry splitEntry) {
        this.splitEntry = splitEntry;
    }

    public GradPair getLeftGradPair() {
        return leftGradPair;
    }

    public GradPair getRightGradPair() {
        return rightGradPair;
    }
}
