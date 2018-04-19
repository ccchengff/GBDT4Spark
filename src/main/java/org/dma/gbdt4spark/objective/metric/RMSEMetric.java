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
    public float eval(float[] preds, float[] labels) {
        double errSum = 0.0f;
        if (preds.length == labels.length) {
            for (int i = 0; i < preds.length; i++) {
                errSum += evalOne(preds[i], labels[i]);
            }
        } else {
            int numLabel = preds.length / labels.length;
            float[] pred = new float[numLabel];
            for (int i = 0; i < labels.length; i++) {
                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
                errSum += evalOne(pred, labels[i]);
            }
        }
        return (float) Math.sqrt(errSum / labels.length);
    }

    @Override
    public float evalOne(float pred, float label) {
        float diff = pred - label;
        return diff * diff;
    }

    @Override
    public float evalOne(float[] pred, float label) {
        float err = 0.0f;
        int trueLabel = (int) label;
        for (int i = 0; i < pred.length; i++) {
            float diff = pred[i] - (i == trueLabel ? 1.0f : 0.0f);
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
