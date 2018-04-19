package org.dma.gbdt4spark.objective.loss;

import org.dma.gbdt4spark.objective.metric.EvalMetric;
import org.dma.gbdt4spark.util.Maths;

import javax.inject.Singleton;
import java.util.Arrays;

@Singleton
public class RMSELoss implements BinaryLoss, MultiLoss {
    private static RMSELoss instance;

    private RMSELoss() {}

    @Override
    public Kind getKind() {
        return Kind.RMSE;
    }

    @Override
    public EvalMetric.Kind defaultEvalMetric() {
        return EvalMetric.Kind.RMSE;
    }

    @Override
    public float firOrderGrad(float pred, float label) {
        return pred - label;
    }

    @Override
    public float secOrderGrad(float pred, float label) {
        return 1.0f;
    }

    @Override
    public float secOrderGrad(float pred, float label, float firGrad) {
        return 1.0f;
    }

    @Override
    public float[] firOrderGrad(float[] pred, float label) {
        int numLabel = pred.length;
        int trueLabel = (int) label;
        float[] grad = new float[numLabel];
        for (int i = 0; i < numLabel; i++)
            grad[i] = pred[i] - (trueLabel == i ? 1 : 0);
        return grad;
    }

    @Override
    public float[] secOrderGradDiag(float[] pred, float label) {
        int numLabel = pred.length;
        float[] hess = new float[numLabel];
        Arrays.fill(hess, 1.0f);
        return hess;
    }

    @Override
    public float[] secOrderGradDiag(float[] pred, float label, float[] firGrad) {
        return secOrderGradDiag(pred, label);
    }

    @Override
    public float[] secOrderGradFull(float[] pred, float label) {
        int numLabel = pred.length;
        int size = (numLabel + 1) * numLabel / 2;
        float[] hess = new float[size];
        for (int i = 0; i < numLabel; i++) {
            int t = Maths.indexOfLowerTriangularMatrix(i, i);
            hess[t] = 1.0f;
        }
        return hess;
    }

    @Override
    public float[] secOrderGradFull(float[] pred, float label, float[] firGrad) {
        return secOrderGradFull(pred, label);
    }

    public static RMSELoss getInstance() {
        if (instance == null)
            instance = new RMSELoss();
        return instance;
    }
}
