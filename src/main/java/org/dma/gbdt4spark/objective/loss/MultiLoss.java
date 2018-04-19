package org.dma.gbdt4spark.objective.loss;

public interface MultiLoss extends Loss {
    float[] firOrderGrad(float[] pred, float label);

    float[] secOrderGradDiag(float[] pred, float label);

    float[] secOrderGradDiag(float[] pred, float label, float[] firGrad);

    float[] secOrderGradFull(float[] pred, float label);

    float[] secOrderGradFull(float[] pred, float label, float[] firGrad);
}
