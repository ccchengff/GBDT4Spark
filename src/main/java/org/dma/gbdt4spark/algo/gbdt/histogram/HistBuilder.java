package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.data.FeatureRow;
import org.dma.gbdt4spark.tree.param.GBDTParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.*;

public class HistBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(HistBuilder.class);

    private final GBDTParam param;

    public HistBuilder(final GBDTParam param) {
        this.param = param;
    }

    public Histogram[] buildHistograms(
            FeatureRow[] featureRows, int nodeStart, int nodeEnd, int[] insPos,
            GradPair[] gradPairs, GradPair sumGradPair, int[] defaultBins) throws Exception {
        Histogram[] histograms = new Histogram[featureRows.length];
        if (param.numThread > 1) {
            ExecutorService threadPool = Executors.newFixedThreadPool(param.numThread);
            Future[] futures = new Future[param.numThread];
            for (int threadId = 0; threadId < param.numThread; threadId++) {
                futures[threadId] = threadPool.submit(new FPBuilderThread(
                        threadId, histograms, featureRows,
                        nodeStart, nodeEnd, insPos, gradPairs, sumGradPair, defaultBins));
            }
            threadPool.shutdown();
            for (Future future : futures)
                future.get();
        } else {
            new FPBuilderThread(0, histograms, featureRows,
                    nodeStart, nodeEnd, insPos, gradPairs, sumGradPair, defaultBins).call();
        }
        return histograms;
    }

    private class FPBuilderThread implements Callable<Void> {
        private final int threadId;
        private final Histogram[] histograms;
        private final FeatureRow[] featureRows;
        private final int nodeStart;
        private final int nodeEnd;
        private final int[] insPos;
        private final GradPair[] gradPairs;
        private final GradPair sumGradPair;
        private final int[] defaultBins;

        private FPBuilderThread(int threadId, Histogram[] histograms, FeatureRow[] featureRows,
                                int nodeStart, int nodeEnd, int[] insPos,
                                GradPair[] gradPairs, GradPair sumGradPair, int[] defaultBins) {
            this.threadId = threadId;
            this.histograms = histograms;
            this.featureRows = featureRows;
            this.nodeStart = nodeStart;
            this.nodeEnd = nodeEnd;
            this.insPos = insPos;
            this.gradPairs = gradPairs;
            this.sumGradPair = sumGradPair;
            this.defaultBins = defaultBins;
        }

        @Override
        public Void call() throws Exception {
            int avg = featureRows.length / param.numThread;
            int from = threadId * avg;
            int to = threadId + 1 == param.numThread ? featureRows.length : from + avg;

            for (int i = from; i < to; i++) {
                if (featureRows[i] == null) continue;
                int[] indices = featureRows[i].indices();
                int[] bins = featureRows[i].bins();
                int nnz = indices.length;
                // 1. allocate histogram
                Histogram hist = new Histogram(param.numSplit, param.numClass, param.fullHessian);
                // 2. loop non-zero instances, plusBy to histogram, and record the gradients taken
                for (int j = 0; j < nnz; j++) {
                    int insId = indices[j];
                    if (nodeStart <= insPos[insId] && insPos[insId] <= nodeEnd) {
                        int binId = bins[j];
                        if (gradPairs[insId] == null) {
                            throw new RuntimeException(String.format("Ins[%d] grad pair is null", insId));
                        }
                        hist.accumulate(binId, gradPairs[insId]);
                    }
                }
                // 3. add remaining grad and hess to default bin
                GradPair taken = hist.sum();
                GradPair remain = sumGradPair.subtract(taken);
                int defaultBinId = defaultBins[i];
                hist.accumulate(defaultBinId, remain);
                // 4. put it to result
                histograms[i] = hist;
            }

            return null;
        }
    }
}
