package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.algo.gbdt.metadata.DataInfo;
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo;
import org.dma.gbdt4spark.data.FeatureRow;
import org.dma.gbdt4spark.tree.param.GBDTParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Option;

import java.util.concurrent.*;

public class HistBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(HistBuilder.class);

    private final GBDTParam param;

    public HistBuilder(final GBDTParam param) {
        this.param = param;
    }

    public Option<Histogram>[] buildHistograms(int[] sampleFeats, int featLo, Option<FeatureRow>[] featureRows,
                                               FeatureInfo featureInfo, DataInfo dataInfo,
                                               int nid, GradPair sumGradPair) throws Exception {
        Option<Histogram>[] histograms = new Option[sampleFeats.length];
        int nodeStart = dataInfo.getNodePosStart(nid);
        int nodeEnd = dataInfo.getNodePosEnd(nid);
        int[] insPos = dataInfo.insPos();
        GradPair[] gradPairs = dataInfo.gradPairs();
        if (param.numThread > 1) {
            ExecutorService threadPool = Executors.newFixedThreadPool(param.numThread);
            Future[] futures = new Future[param.numThread];
            for (int threadId = 0; threadId < param.numThread; threadId++) {
                futures[threadId] = threadPool.submit(new BuilderThread(threadId, sampleFeats, featLo,
                        featureRows, featureInfo, nodeStart, nodeEnd, insPos,
                        gradPairs, sumGradPair, histograms));
            }
            threadPool.shutdown();
            for (Future future : futures)
                future.get();
        } else {
            new BuilderThread(0, sampleFeats, featLo, featureRows, featureInfo,
                    nodeStart, nodeEnd, insPos, gradPairs, sumGradPair, histograms).call();
        }
        return histograms;
    }

    private class BuilderThread implements Callable<Void> {
        private final int threadId;
        private final int[] sampleFeats;
        private final int featLo;
        private final Option<FeatureRow>[] featureRows;
        private final FeatureInfo featureInfo;
        private final int nodeStart;
        private final int nodeEnd;
        private final int[] insPos;
        private final GradPair[] gradPairs;
        private final GradPair sumGradPair;
        private final Option<Histogram>[] histograms;

        private BuilderThread(int threadId, int[] sampleFeats, int featLo,
                              Option<FeatureRow>[] featureRows, FeatureInfo featureInfo,
                              int nodeStart, int nodeEnd, int[] insPos, GradPair[] gradPairs,
                              GradPair sumGradPair, Option<Histogram>[] histograms) {
            this.threadId = threadId;
            this.sampleFeats = sampleFeats;
            this.featLo = featLo;
            this.featureRows = featureRows;
            this.featureInfo = featureInfo;
            this.nodeStart = nodeStart;
            this.nodeEnd = nodeEnd;
            this.insPos = insPos;
            this.gradPairs = gradPairs;
            this.sumGradPair = sumGradPair;
            this.histograms = histograms;
        }

        @Override
        public Void call() throws Exception {
            int avg = sampleFeats.length / param.numThread;
            int from = threadId * avg;
            int to = threadId + 1 == param.numThread ? featureRows.length : from + avg;

            for (int i = from; i < to; i++) {
                int fid = sampleFeats[i];
                if (featureRows[fid - featLo].isDefined()) {
                    FeatureRow featRow = featureRows[fid - featLo].get();
                    int[] indices = featRow.indices();
                    int[] bins = featRow.bins();
                    int nnz = indices.length;
                    if (nnz != 0) {
                        // 1. allocate histogram
                        int numBin = featureInfo.getNumBin(fid);
                        Histogram hist = new Histogram(numBin, param.numClass, param.fullHessian);
                        // 2. loop non-zero instances, accumulate to histogram
                        // TODO binary search
                        for (int j = 0; j < nnz; j++) {
                            int insId = indices[j];
                            if (nodeStart <= insPos[insId] && insPos[insId] <= nodeEnd) {
                                int binId = bins[j];
                                hist.accumulate(binId, gradPairs[insId]);
                            }
                        }
                        // 3. add remaining grad and hess to default bin
                        GradPair taken = hist.sum();
                        GradPair remain = sumGradPair.subtract(taken);
                        int defaultBin = featureInfo.getDefaultBin(fid);
                        hist.accumulate(defaultBin, remain);
                        histograms[i] = Option.apply(hist);
                    } else {
                        histograms[i] = Option.empty();
                    }
                } else {
                    histograms[i] = Option.empty();
                }
            }

            return null;
        }
    }
}
