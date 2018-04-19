package org.dma.gbdt4spark.objective.metric;

import org.dma.gbdt4spark.util.Maths;

import javax.inject.Singleton;

@Singleton
public class PrecisionMetric implements EvalMetric {
    private static PrecisionMetric instance;

    private PrecisionMetric() {}

    @Override
    public Kind getKind() {
        return Kind.PRECISION;
    }

    @Override
    public float eval(float[] preds, float[] labels) {
        float correct = 0.0f;
        if (preds.length == labels.length) {
            for (int i = 0; i < preds.length; i++) {
                correct += evalOne(preds[i], labels[i]);
            }
        } else {
            int numLabel = preds.length / labels.length;
            float[] pred = new float[numLabel];
            for (int i = 0; i < labels.length; i++) {
                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
                correct += evalOne(pred, labels[i]);
            }
        }
        return correct / labels.length;
    }

    @Override
    public float evalOne(float pred, float label) {
        return pred < 0.0f ? 1.0f - label : label;
    }

    @Override
    public float evalOne(float[] pred, float label) {
        return Maths.argmax(pred) == ((int) label) ? 1.0f : 0.0f;
    }

    public static PrecisionMetric getInstance() {
        if (instance == null)
            instance = new PrecisionMetric();
        return instance;
    }
}
