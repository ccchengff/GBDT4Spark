package org.dma.gbdt4spark.algo.gbdt.histogram;

import java.io.Serializable;

public class Histogram implements Serializable {
    private int numBin;
    private GradPair[] histogram;

    public Histogram(int numBin, GradPair[] histogram) {
        this.numBin = numBin;
        this.histogram = histogram;
    }

    public Histogram(int numBin, int numClass, boolean fullHessian) {
        this.numBin = numBin;
        this.histogram = new GradPair[numBin];
        if (numClass == 2) {
            for (int i = 0; i < numBin; i++) {
                this.histogram[i] = new BinaryGradPair();
            }
        } else {
            for (int i = 0; i < numBin; i++) {
                this.histogram[i] = new MultiGradPair(numClass, fullHessian);
            }
        }
    }

    public void accumulate(int index, GradPair gradPair) {
        histogram[index].plusBy(gradPair);
    }

    public Histogram plus(Histogram other) {
        GradPair[] res = new GradPair[numBin];
        GradPair[] h1 = this.histogram;
        GradPair[] h2 = other.histogram;
        for (int i = 0; i < numBin; i++) {
            res[i] = h1[i].plus(h2[i]);
        }
        return new Histogram(numBin, res);
    }

    public Histogram subtract(Histogram other) {
        GradPair[] res = new GradPair[numBin];
        GradPair[] h1 = this.histogram;
        GradPair[] h2 = other.histogram;
        for (int i = 0; i < numBin; i++) {
            res[i] = h1[i].subtract(h2[i]);
        }
        return new Histogram(numBin, res);
    }

    public void plusBy(Histogram other) {
        for (int i = 0; i < numBin; i++) {
            this.histogram[i].plusBy(other.histogram[i]);
        }
    }

    public void subtractBy(Histogram other) {
        for (int i = 0; i < numBin; i++) {
            this.histogram[i].subtractBy(other.histogram[i]);
        }
    }

    public GradPair sum() {
        return sum(0, numBin);
    }

    public GradPair sum(int start, int end) {
        GradPair sum = histogram[start].copy();
        for (int i = start + 1; i < end; i++) {
            sum.plusBy(histogram[i]);
        }
        return sum;
    }

    public int getNumBin() {
        return numBin;
    }

    public GradPair get(int index) {
        return histogram[index];
    }

    public GradPair[] getHistogram() {
        return histogram;
    }
}
