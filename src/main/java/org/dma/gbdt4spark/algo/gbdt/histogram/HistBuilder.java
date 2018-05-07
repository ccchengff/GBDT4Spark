package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.algo.gbdt.metadata.DataInfo;
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo;
import org.dma.gbdt4spark.data.FeatureRow;
import org.dma.gbdt4spark.tree.param.GBDTParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Option;

import java.util.Arrays;
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
        if (param.numThread > 1) {
            ExecutorService threadPool = Executors.newFixedThreadPool(param.numThread);
            Future[] futures = new Future[param.numThread];
            for (int threadId = 0; threadId < param.numThread; threadId++) {
                futures[threadId] = threadPool.submit(new BuilderThread(threadId, sampleFeats, featLo,
                        featureRows, featureInfo, dataInfo, nid, sumGradPair, histograms));
            }
            threadPool.shutdown();
            for (Future future : futures)
                future.get();
        } else {
            new BuilderThread(0, sampleFeats, featLo, featureRows, featureInfo,
                    dataInfo, nid, sumGradPair, histograms).call();
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
        private final int[] nodeToIns;
        private final int[] insPos;
        private final GradPair[] gradPairs;
        private final GradPair sumGradPair;
        private final Option<Histogram>[] histograms;

        private BuilderThread(int threadId, int[] sampleFeats, int featLo,
                              Option<FeatureRow>[] featureRows, FeatureInfo featureInfo,
                              DataInfo dataInfo, int nid, GradPair sumGradPair,
                              Option<Histogram>[] histograms) {
            this.threadId = threadId;
            this.sampleFeats = sampleFeats;
            this.featLo = featLo;
            this.featureRows = featureRows;
            this.featureInfo = featureInfo;
            this.nodeStart = dataInfo.getNodePosStart(nid);
            this.nodeEnd = dataInfo.getNodePosEnd(nid);
            this.nodeToIns = dataInfo.nodeToIns();
            this.insPos = dataInfo.insPos();
            this.gradPairs = dataInfo.gradPairs();
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
                        if (nnz <= nodeEnd - nodeStart + 1) { // loop all nnz of current feature
                            for (int j = 0; j < nnz; j++) {
                                int insId = indices[j];
                                if (nodeStart <= insPos[insId] && insPos[insId] <= nodeEnd) {
                                    int binId = bins[j];
                                    hist.accumulate(binId, gradPairs[insId]);
                                }
                            }
                        } else { // for all instance on this node, binary search in feature row
                            for (int j = nodeStart; j <= nodeEnd; j++) {
                                int insId = nodeToIns[j];
                                int index = Arrays.binarySearch(indices, insId);
                                if (index >= 0) {
                                    int binId = bins[index];
                                    hist.accumulate(binId, gradPairs[insId]);
                                }
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
