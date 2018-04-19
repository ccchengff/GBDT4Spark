package org.dma.gbdt4spark.objective.loss;

public interface BinaryLoss extends Loss {
    float firOrderGrad(float pred, float label);

    float secOrderGrad(float pred, float label);

    float secOrderGrad(float pred, float label, float firGrad);
}
