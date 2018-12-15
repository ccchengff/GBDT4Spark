package org.dma.gbdt4spark.objective.metric;

import javax.inject.Singleton;

@Singleton
public class RMSEMetric implements EvalMetric {
    private static RMSEMetric instance;

    private RMSEMetric() {}

    @Override
    public Kind getKind() {
        return Kind.RMSE;
    }

    @Override
    public double sum(float[] preds, float[] labels) {
        return sum(preds, labels, 0, labels.length);
    }

    @Override
    public double sum(float[] preds, float[] labels, int start, int end) {
        double errSum = 0.0f;
        if (preds.length == labels.length) {
            for (int i = start; i < end; i++) {
                errSum += evalOne(preds[i], labels[i]);
            }
        } else {
            int numLabel = preds.length / labels.length;
            float[] pred = new float[numLabel];
            for (int i = start; i < end; i++) {
                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
                errSum += evalOne(pred, labels[i]);
            }
        }
        return errSum;
    }

    @Override
    public double avg(double sum, int num) {
        return Math.sqrt(sum / num);
    }

    @Override
    public double eval(float[] preds, float[] labels) {
        return avg(sum(preds, labels), labels.length);
//        double errSum = 0.0f;
//        if (preds.length == labels.length) {
//            for (int i = 0; i < preds.length; i++) {
//                errSum += evalOne(preds[i], labels[i]);
//            }
//        } else {
//            int numLabel = preds.length / labels.length;
//            float[] pred = new float[numLabel];
//            for (int i = 0; i < labels.length; i++) {
//                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
//                errSum += evalOne(pred, labels[i]);
//            }
//        }
//        return Math.sqrt(errSum / labels.length);
    }

    @Override
    public double evalOne(float pred, float label) {
        double diff = pred - label;
        return diff * diff;
    }

    @Override
    public double evalOne(float[] pred, float label) {
        double err = 0.0;
        int trueLabel = (int) label;
        for (int i = 0; i < pred.length; i++) {
            double diff = pred[i] - (i == trueLabel ? 1 : 0);
            err += diff * diff;
        }
        return err;
    }

    public static RMSEMetric getInstance() {
        if (instance == null)
            instance = new RMSEMetric();
        return instance;
    }
}
