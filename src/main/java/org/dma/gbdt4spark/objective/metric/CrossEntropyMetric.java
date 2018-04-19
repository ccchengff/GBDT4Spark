package org.dma.gbdt4spark.objective.metric;

import com.google.common.base.Preconditions;
import org.dma.gbdt4spark.exception.GBDTException;
import org.dma.gbdt4spark.util.Maths;

import javax.inject.Singleton;

@Singleton
public class CrossEntropyMetric implements EvalMetric {
    private static CrossEntropyMetric instance;

    private CrossEntropyMetric() {}

    @Override
    public Kind getKind() {
        return Kind.CROSS_ENTROPY;
    }

    @Override
    public float eval(float[] preds, float[] labels) {
        Preconditions.checkArgument(preds.length != labels.length
                        && preds.length % labels.length == 0,
                "CrossEntropyMetric should be used for multi-label classification");
        float loss = 0.0f;
        int numLabel = preds.length / labels.length;
        float[] pred = new float[numLabel];
        for (int i = 0; i < labels.length; i++) {
            System.arraycopy(preds, i * numLabel, pred, 0, numLabel);
            loss += evalOne(pred, labels[i]);
        }
        return loss / labels.length;
    }

    @Override
    public float evalOne(float pred, float label) {
        throw new GBDTException("CrossEntropyMetric should be used for multi-label classification");
    }

    @Override
    public float evalOne(float[] pred, float label) {
        float sum = 0.0f;
        for (float p : pred) {
            sum += Math.exp(p);
        }
        float p = (float) Math.exp(pred[(int) label]) / sum;
        return (float) -Math.log(Math.max(p, Maths.EPSILON));
    }

    public static CrossEntropyMetric getInstance() {
        if (instance == null)
            instance = new CrossEntropyMetric();
        return instance;
    }
}
