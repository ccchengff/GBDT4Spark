package org.dma.gbdt4spark.objective.loss;

public interface BinaryLoss extends Loss {
    double firOrderGrad(float pred, float label);

    double secOrderGrad(float pred, float label);

    double secOrderGrad(float pred, float label, double firGrad);
}
