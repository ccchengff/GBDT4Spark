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
    public double sum(float[] preds, float[] labels) {
        return sum(preds, labels, 0, labels.length);
    }

    @Override
    public double sum(float[] preds, float[] labels, int start, int end) {
        double correct = 0.0;
        if (preds.length == labels.length) {
            for (int i = start; i < end; i++) {
                correct += evalOne(preds[i], labels[i]);
            }
        } else {
            int numLabel = preds.length / labels.length;
            float[] pred = new float[numLabel];
            for (int i = start; i < end; i++) {
                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
                correct += evalOne(pred, labels[i]);
            }
        }
        return correct;
    }

    @Override
    public double avg(double sum, int num) {
        return sum / num;
    }

    @Override
    public double eval(float[] preds, float[] labels) {
        return avg(sum(preds, labels), labels.length);
//        double correct = 0.0;
//        if (preds.length == labels.length) {
//            for (int i = 0; i < preds.length; i++) {
//                correct += evalOne(preds[i], labels[i]);
//            }
//        } else {
//            int numLabel = preds.length / labels.length;
//            float[] pred = new float[numLabel];
//            for (int i = 0; i < labels.length; i++) {
//                System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
//                correct += evalOne(pred, labels[i]);
//            }
//        }
//        return (float) (correct / labels.length);
    }

    @Override
    public double evalOne(float pred, float label) {
        return pred < 0.0f ? 1.0 - label : label;
    }

    @Override
    public double evalOne(float[] pred, float label) {
        return Maths.argmax(pred) == ((int) label) ? 1 : 0;
    }

    public static PrecisionMetric getInstance() {
        if (instance == null)
            instance = new PrecisionMetric();
        return instance;
    }
}
