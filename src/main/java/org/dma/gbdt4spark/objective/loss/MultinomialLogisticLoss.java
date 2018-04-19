package org.dma.gbdt4spark.objective.loss;

import org.dma.gbdt4spark.objective.metric.EvalMetric;
import org.dma.gbdt4spark.util.Maths;

import javax.inject.Singleton;

@Singleton
public class MultinomialLogisticLoss implements MultiLoss {
    private static MultinomialLogisticLoss instance;

    private MultinomialLogisticLoss() {}

    @Override
    public Kind getKind() {
        return Kind.MultiLogistic;
    }

    @Override
    public EvalMetric.Kind defaultEvalMetric() {
        return EvalMetric.Kind.CROSS_ENTROPY;
    }

    @Override
    public float[] firOrderGrad(float[] pred, float label) {
        float[] prob = pred.clone();
        Maths.softmax(prob);
        int trueLabel = (int) label;
        float[] grad = prob;
        for (int i = 0; i < grad.length; i++) {
            grad[i] = (trueLabel == i ? prob[i] - 1.0f : prob[i]);
        }
        return grad;
    }

    @Override
    public float[] secOrderGradDiag(float[] pred, float label) {
        float[] prob = pred.clone();
        Maths.softmax(prob);
        float[] hess = prob;
        for (int i = 0; i < hess.length; i++) {
            hess[i] = Math.max(prob[i] * (1.0f - prob[i]), Maths.EPSILON);
        }
        return hess;
    }

    @Override
    public float[] secOrderGradDiag(float[] pred, float label, float[] firGrad) {
        int trueLabel = (int) label;
        float[] hess = new float[pred.length];
        for (int i = 0; i < hess.length; i++) {
            float prob = trueLabel == i ? firGrad[i] + 1.0f : firGrad[i];
            hess[i] = Math.max(prob * (1.0f - prob), Maths.EPSILON);
        }
        return hess;
    }

    @Override
    public float[] secOrderGradFull(float[] pred, float label) {
        float[] prob = pred.clone();
        Maths.softmax(prob);
        int numLabel = pred.length;
        float[] hess = new float[numLabel * (numLabel + 1) / 2];
        for (int i = 0; i < numLabel; i++) {
            int rowI = Maths.indexOfLowerTriangularMatrix(i, 0);
            for (int j = 0; j < i; j++) {
                hess[rowI + j] = Math.min(-prob[i] * prob[j], -Maths.EPSILON);
            }
            hess[rowI + i] = Math.max(prob[i] * (1.0f - prob[i]), Maths.EPSILON);
        }
        return hess;
    }

    @Override
    public float[] secOrderGradFull(float[] pred, float label, float[] firGrad) {
        int numLabel = pred.length;
        int trueLabel = (int) label;
        float[] prob = new float[numLabel];
        for (int i = 0; i < numLabel; i++)
            prob[i] = trueLabel == i ? firGrad[i] + 1.0f : firGrad[i];
        float[] hess = new float[numLabel * (numLabel + 1) / 2];
        for (int i = 0; i < numLabel; i++) {
            int rowI = Maths.indexOfLowerTriangularMatrix(i, 0);
            for (int j = 0; j < i; j++) {
                hess[rowI + j] = Math.min(-prob[i] * prob[j], -Maths.EPSILON);
            }
            hess[rowI + i] = Math.max(prob[i] * (1.0f - prob[i]), Maths.EPSILON);
        }
        return hess;
    }

    public static MultinomialLogisticLoss getInstance() {
        if (instance == null)
            instance = new MultinomialLogisticLoss();
        return instance;
    }
}
