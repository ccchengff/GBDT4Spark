package org.dma.gbdt4spark.algo.gbdt.histogram;

import com.google.common.base.Preconditions;
import org.dma.gbdt4spark.algo.gbdt.metadata.DataInfo;
import org.dma.gbdt4spark.algo.gbdt.metadata.FeatureInfo;
import org.dma.gbdt4spark.data.FeatureRow;
import org.dma.gbdt4spark.data.InstanceRow;
import org.dma.gbdt4spark.tree.param.GBDTParam;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
import org.dma.gbdt4spark.logging.Logger;
import org.dma.gbdt4spark.logging.LoggerFactory;
import scala.Option;

import java.util.Arrays;
import java.util.concurrent.*;

public class HistBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(HistBuilder.class);

    private final GBDTParam param;

    private static int MIN_INSTANCE_PER_THREAD = 10000;
    private ExecutorService threadPool;
    private FPBuilderThread[] fpThreads;

    public HistBuilder(final GBDTParam param) {
        this.param = param;
        if (param.numThread > 1) {
            this.threadPool = Executors.newFixedThreadPool(param.numThread);
            this.fpThreads = new FPBuilderThread[param.numThread];
            for (int threadId = 0; threadId < param.numThread; threadId++) {
                this.fpThreads[threadId] = new FPBuilderThread(threadId, param);
            }
        }
    }

    public void shutdown() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
    }

    private static class FPBuilderThread implements Callable<Histogram[]> {
        int threadId;
        GBDTParam param;
        boolean[] isFeatUsed;
        int featLo;
        FeatureInfo featureInfo;
        InstanceRow[] instanceRows;
        GradPair[] gradPairs;
        int[] nodeToIns;
        int from, to;

        public FPBuilderThread(int threadId, GBDTParam param) {
            this.threadId = threadId;
            this.param = param;
        }

        @Override
        public Histogram[] call() throws Exception {
            return sparseBuildFP(param, isFeatUsed, featLo, featureInfo, instanceRows,
                    gradPairs, nodeToIns, from, to);
        }
    }


    private static Histogram[] sparseBuildFP(GBDTParam param, boolean[] isFeatUsed,
                                             int featLo, FeatureInfo featureInfo,
                                             InstanceRow[] instanceRows, GradPair[] gradPairs,
                                             int[] nodeToIns, int from, int to) {
        Histogram[] histograms = new Histogram[isFeatUsed.length];
        for (int i = 0; i < isFeatUsed.length; i++) {
            if (isFeatUsed[i])
                histograms[i] = new Histogram(featureInfo.getNumBin(featLo + i),
                        param.numClass, param.fullHessian);
        }
        for (int posId = from; posId < to; posId++) {
            int insId = nodeToIns[posId];
            InstanceRow ins = instanceRows[insId];
            int[] indices = ins.indices();
            int[] bins = ins.bins();
            int nnz = indices.length;
            for (int j = 0; j < nnz; j++) {
                int fid = indices[j];
                if (isFeatUsed[fid - featLo]) {
                    histograms[fid - featLo].accumulate(bins[j], gradPairs[insId]);
                }
            }
        }
        return histograms;
    }

    public Histogram[] buildHistogramsFP(boolean[] isFeatUsed, int featLo, InstanceRow[] instanceRows,
                                         FeatureInfo featureInfo, DataInfo dataInfo,
                                         int nid, GradPair sumGradPair) throws Exception {
        int nodeStart = dataInfo.getNodePosStart(nid);
        int nodeEnd = dataInfo.getNodePosEnd(nid);
        GradPair[] gradPairs = dataInfo.gradPairs();
        int[] nodeToIns = dataInfo.nodeToIns();

        Histogram[] res;
        if (param.numThread <= 1 || nodeEnd - nodeStart + 1 <= MIN_INSTANCE_PER_THREAD) {
            res = sparseBuildFP(param, isFeatUsed, featLo, featureInfo, instanceRows,
                    gradPairs, nodeToIns, nodeStart, nodeEnd);
        } else {
            int actualNumThread = Math.min(param.numThread,
                    (nodeEnd - nodeStart + 1 + MIN_INSTANCE_PER_THREAD - 1) / MIN_INSTANCE_PER_THREAD);
            LOG.info(String.format("Number of instances[%d], pos[%d, %d] actual thread num[%d]",
                    nodeEnd - nodeStart + 1, nodeStart, nodeEnd, actualNumThread));
            Future[] futures = new Future[actualNumThread];
            int avg = (nodeEnd - nodeStart + 1) / actualNumThread;
            int from = nodeStart, to = nodeStart + avg;
            for (int threadId = 0; threadId < actualNumThread; threadId++) {
                LOG.info(String.format("Thread[%d] from[%d] to[%d]",
                        threadId, from, to));
                FPBuilderThread builder = fpThreads[threadId];
                builder.isFeatUsed = isFeatUsed;
                builder.featLo = featLo;
                builder.featureInfo = featureInfo;
                builder.instanceRows = instanceRows;
                builder.gradPairs = gradPairs;
                builder.nodeToIns = nodeToIns;
                builder.from = from;
                builder.to = to;
                from = to;
                to = Math.min(from + avg, nodeEnd + 1);
                futures[threadId] = threadPool.submit(builder);
            }
            res = (Histogram[]) futures[0].get();
            for (int threadId = 1; threadId < actualNumThread; threadId++) {
                Histogram[] hist = (Histogram[]) futures[threadId].get();
                for (int i = 0; i < res.length; i++)
                    if (res[i] != null)
                        res[i].plusBy(hist[i]);
            }
        }
        for (int i = 0; i < res.length; i++) {
            if (res[i] != null) {
                GradPair taken = res[i].sum();
                GradPair remain = sumGradPair.subtract(taken);
                int defaultBin = featureInfo.getDefaultBin(featLo + i);
                res[i].accumulate(defaultBin, remain);
            }
        }
        return res;
    }

    public Histogram[] histSubtraction(Histogram[] mined, Histogram[] miner, boolean inPlace) {
        if (inPlace) {
            for (int i = 0; i < mined.length; i++) {
                if (mined[i] != null)
                    mined[i].subtractBy(miner[i]);
            }
            return mined;
        } else {
            Histogram[] res = new Histogram[mined.length];
            for (int i = 0; i < mined.length; i++) {
                if (mined[i] != null)
                    res[i] = mined[i].subtract(miner[i]);
            }
            return res;
        }
    }

    public Option<Histogram>[] buildHistogram(int[] sampleFeats, int featLo, FeatureInfo featureInfo, DataInfo dataInfo,
                                               InstanceRow[] instanceRows, int nid, GradPair sumGradPair) {
        int numF = sampleFeats.length;
        Histogram[] histograms = new Histogram[numF];
        for (int i = 0; i < numF; i++) {
            int fid = sampleFeats[i];
            if (featureInfo.nnz(fid) > 0)
                histograms[i] = new Histogram(featureInfo.getNumBin(fid),
                        param.numClass, param.fullHessian);
        }
        GradPair[] gradPairs = dataInfo.gradPairs();
        int nodeStart = dataInfo.getNodePosStart(nid);
        int nodeEnd = dataInfo.getNodePosEnd(nid);
        int[] nodeToPos = dataInfo.nodeToIns();
        for (int posId = nodeStart; posId <= nodeEnd; posId++) {
            int insId = nodeToPos[posId];
            if (instanceRows[insId] != null) {
                int[] indices = instanceRows[insId].indices();
                int[] bins = instanceRows[insId].bins();
                int nnz = indices.length;
                for (int j = 0; j < nnz; j++) {
                    histograms[indices[j] - featLo].accumulate(bins[j], gradPairs[insId]);
                }
            }
        }
        Option<Histogram>[] res = new Option[numF];
        for (int i = 0; i < numF; i++) {
            if (histograms[i] != null) {
                GradPair taken = histograms[i].sum();
                GradPair remain = sumGradPair.subtract(taken);
                int defaultBin = featureInfo.getDefaultBin(featLo + i);
                histograms[i].accumulate(defaultBin, remain);
                res[i] = Option.apply(histograms[i]);
            } else {
                res[i] = Option.empty();
            }
        }
        return res;
    }

    public Option<Histogram>[] buildHistograms(int[] sampleFeats, int featLo, Option<FeatureRow>[] featureRows,
                                               FeatureInfo featureInfo, DataInfo dataInfo, InstanceRow[] instanceRows,
                                               int nid, GradPair sumGradPair) throws Exception {
        Preconditions.checkArgument(sampleFeats.length == featureRows.length);
        int numFeat = sampleFeats.length;
        Histogram[] histograms = new Histogram[numFeat];
        for (int i = 0; i < numFeat; i++) {
            if (featureRows[i].isDefined()) {
                int nnz = featureRows[i].get().size();
                if (nnz > 0) {
                    int fid = featLo + i;
                    int numBin = featureInfo.getNumBin(fid);
                    histograms[i] = new Histogram(numBin, param.numClass, param.fullHessian);
                }
            }
        }
        GradPair[] gradPairs = dataInfo.gradPairs();
        int nodeStart = dataInfo.getNodePosStart(nid);
        int nodeEnd = dataInfo.getNodePosEnd(nid);
        int[] nodeToPos = dataInfo.nodeToIns();
        for (int posId = nodeStart; posId <= nodeEnd; posId++) {
            int insId = nodeToPos[posId];
            if (instanceRows[insId] != null) {
                int[] indices = instanceRows[insId].indices();
                int[] bins = instanceRows[insId].bins();
                int nnz = indices.length;
                for (int j = 0; j < nnz; j++) {
                    histograms[indices[j] - featLo].accumulate(bins[j], gradPairs[insId]);
                }
            }
        }
        Option<Histogram>[] res = new Option[histograms.length];
        for (int i = 0; i < histograms.length; i++) {
            if (histograms[i] != null) {
                GradPair taken = histograms[i].sum();
                GradPair remain = sumGradPair.subtract(taken);
                int defaultBin = featureInfo.getDefaultBin(featLo + i);
                histograms[i].accumulate(defaultBin, remain);
                res[i] = Option.apply(histograms[i]);
            } else {
                res[i] = Option.empty();
            }
        }
        return res;
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

            long startTime = System.currentTimeMillis();
            long prepareCost = 0L;
            long allocCost = 0L;
            long accCost = 0L;
            long addRemainCost = 0L;
            long foreachCost = 0L;
            long mergeCost = 0L;
            long binarySearchCost = 0L;

            for (int i = from; i < to; i++) {
                int fid = sampleFeats[i];
                if (featureRows[fid - featLo].isDefined()) {
                    long t;
                    t = System.currentTimeMillis();
                    FeatureRow featRow = featureRows[fid - featLo].get();
                    int[] indices = featRow.indices();
                    int[] bins = featRow.bins();
                    int nnz = indices.length;
                    prepareCost += System.currentTimeMillis() - t;
                    if (nnz != 0) {
                        // 1. allocate histogram
                        t = System.currentTimeMillis();
                        int numBin = featureInfo.getNumBin(fid);
                        Histogram hist = new Histogram(numBin, param.numClass, param.fullHessian);
                        allocCost += System.currentTimeMillis() - t;
                        // 2. loop non-zero instances, accumulate to histogram
                        if (true) {
                        //if (nnz <= nodeEnd - nodeStart + 1) { // loop all nnz of current feature
                            long t2 = System.currentTimeMillis();
                            for (int j = 0; j < nnz; j++) {
                                int insId = indices[j];
                                if (nodeStart <= insPos[insId] && insPos[insId] <= nodeEnd) {
                                    int binId = bins[j];
                                    t = System.currentTimeMillis();
                                    hist.accumulate(binId, gradPairs[insId]);
                                    accCost += System.currentTimeMillis() - t;
                                }
                            }
                            foreachCost += System.currentTimeMillis() - t2;
                        } else { // for all instance on this node, binary search in feature row
                            long t2 = System.currentTimeMillis();
                            for (int j = nodeStart; j <= nodeEnd; j++) {
                                int insId = nodeToIns[j];
                                int index = Arrays.binarySearch(indices, insId);
                                if (index >= 0) {
                                    int binId = bins[index];
                                    t = System.currentTimeMillis();
                                    hist.accumulate(binId, gradPairs[insId]);
                                    accCost += System.currentTimeMillis() - t;
                                }
                            }
                            binarySearchCost += System.currentTimeMillis() - t2;
                        }
                        // 3. add remaining grad and hess to default bin
                        t = System.currentTimeMillis();
                        GradPair taken = hist.sum();
                        GradPair remain = sumGradPair.subtract(taken);
                        int defaultBin = featureInfo.getDefaultBin(fid);
                        addRemainCost += System.currentTimeMillis() - t;
                        t = System.currentTimeMillis();
                        hist.accumulate(defaultBin, remain);
                        accCost += System.currentTimeMillis() - t;
                        histograms[i] = Option.apply(hist);
                    } else {
                        histograms[i] = Option.empty();
                    }
                } else {
                    histograms[i] = Option.empty();
                }
            }

            //LOG.info(String.format("Build hist cost %d ms, prepare[%d], alloc[%d], acc[%d], addRemain[%d]," +
            //        "foreach[%d], binarySearch[%d]", System.currentTimeMillis() - startTime,
            //        prepareCost, allocCost, accCost, addRemainCost, foreachCost, binarySearchCost));

            LOG.info(String.format("Build hist cost %d ms, prepare[%d], alloc[%d], acc[%d], addRemain[%d]," +
                            "merge[%d]", System.currentTimeMillis() - startTime,
                    prepareCost, allocCost, accCost, addRemainCost, mergeCost));

            return null;
        }
    }
}
