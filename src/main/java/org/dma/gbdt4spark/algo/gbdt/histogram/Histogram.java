package org.dma.gbdt4spark.algo.gbdt.histogram;

import java.io.Serializable;

public class Histogram implements Serializable {
    private int numSplit;
    private GradPair[] histogram;

    public Histogram(int numSplit, GradPair[] histogram) {
        this.numSplit = numSplit;
        this.histogram = histogram;
    }

    public Histogram(int numSplit, int numClass, boolean fullHessian) {
        this.numSplit = numSplit;
        this.histogram = new GradPair[numSplit];
        if (numClass == 2) {
            for (int i = 0; i < numSplit; i++) {
                this.histogram[i] = new BinaryGradPair();
            }
        } else {
            for (int i = 0; i < numSplit; i++) {
                this.histogram[i] = new MultiGradPair(numClass, fullHessian);
            }
        }
    }

    public void accumulate(int index, GradPair gradPair) {
        histogram[index].plusBy(gradPair);
    }

    public Histogram plus(Histogram other) {
        GradPair[] res = new GradPair[numSplit];
        GradPair[] h1 = this.histogram;
        GradPair[] h2 = other.histogram;
        for (int i = 0; i < numSplit; i++) {
            res[i] = h1[i].plus(h2[i]);
        }
        return new Histogram(numSplit, res);
    }

    public Histogram subtract(Histogram other) {
        GradPair[] res = new GradPair[numSplit];
        GradPair[] h1 = this.histogram;
        GradPair[] h2 = other.histogram;
        for (int i = 0; i < numSplit; i++) {
            res[i] = h1[i].subtract(h2[i]);
        }
        return new Histogram(numSplit, res);
    }

    public void plusBy(Histogram other) {
        for (int i = 0; i < numSplit; i++) {
            this.histogram[i].plusBy(other.histogram[i]);
        }
    }

    public void subtractBy(Histogram other) {
        for (int i = 0; i < numSplit; i++) {
            this.histogram[i].subtractBy(other.histogram[i]);
        }
    }

    public GradPair sum() {
        return sum(0, numSplit);
    }

    public GradPair sum(int start, int end) {
        GradPair sum = histogram[start].copy();
        for (int i = start + 1; i < end; i++) {
            sum.plusBy(histogram[i]);
        }
        return sum;
    }

    public int getNumSplit() {
        return numSplit;
    }

    public GradPair get(int index) {
        return histogram[index];
    }

    public GradPair[] getHistogram() {
        return histogram;
    }
}
