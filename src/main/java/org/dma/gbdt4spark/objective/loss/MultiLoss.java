package org.dma.gbdt4spark.objective.loss;

public interface MultiLoss extends Loss {
    double[] firOrderGrad(float[] pred, float label);

    double[] secOrderGradDiag(float[] pred, float label);

    double[] secOrderGradDiag(float[] pred, float label, double[] firGrad);

    double[] secOrderGradFull(float[] pred, float label);

    double[] secOrderGradFull(float[] pred, float label, double[] firGrad);
}
