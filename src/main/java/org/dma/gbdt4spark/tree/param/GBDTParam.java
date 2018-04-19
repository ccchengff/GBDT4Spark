package org.dma.gbdt4spark.tree.param;

import org.dma.gbdt4spark.util.Maths;

public class GBDTParam extends RegTParam {
    public int numClass; // number of classes/labels
    public int numTree;  // number of trees
    //public boolean leafwise;  // true if leaf-wise training, false if level-wise training
    public int numThread;  // parallelism
    //public int batchNum;  // number of batch in mini-batch histogram building

    public boolean fullHessian;  // whether to use full hessian matrix instead of diagonal
    public float minChildWeight;  // minimum amount of hessian (weight) allowed for a child
    public float regAlpha;  // L1 regularization factor
    public float regLambda;  // L2 regularization factor
    public float maxLeafWeight; // maximum leaf weight, default 0 means no constraints

    public String lossFunc; // name of loss function
    public String[] evalMetrics; // name of eval metric

    /**
     * Whether the sum of hessian satisfies weight
     *
     * @param sumHess sum of hessian values
     * @return true if satisfied, false otherwise
     */
    public boolean satisfyWeight(float sumHess) {
        return sumHess >= minChildWeight;
    }

    /**
     * Whether the sum of hessian satisfies weight
     * Since hessian matrix is positive, we have det(hess) <= a11*a22*...*akk,
     * thus we approximate det(hess) with a11*a22*...*akk
     *
     * @param sumHess sum of hessian values
     * @return true if satisfied, false otherwise
     */
    public boolean satisfyWeight(float[] sumHess) {
        if (minChildWeight == 0.0f) return true;
        float w = 1.0f;
        if (!fullHessian) {
            for (float h : sumHess) w *= h;
        } else {
            for (int k = 0; k < numClass; k++) {
                int index = Maths.indexOfLowerTriangularMatrix(k, k);
                w *= sumHess[index];
            }
        }
        return w >= minChildWeight;
    }

    /**
     * Calculate leaf weight given the statistics
     *
     * @param sumGrad sum of gradient values
     * @param sumHess sum of hessian values
     * @return weight
     */
    public float calcWeight(float sumGrad, float sumHess) {
        if (!satisfyWeight(sumHess) || sumGrad == 0.0f) {
            return 0.0f;
        }
        float dw;
        if (regAlpha == 0.0f) {
            dw = -sumGrad / (sumHess + regLambda);
        } else {
            dw = -Maths.thresholdL1(sumGrad, regAlpha) / (sumHess + regLambda);
        }
        if (maxLeafWeight != 0.0f) {
            if (dw > maxLeafWeight) {
                dw = maxLeafWeight;
            } else if (dw < -maxLeafWeight) {
                dw = -maxLeafWeight;
            }
        }
        return dw;
    }

    public float[] calcWeights(float[] sumGrad, float[] sumHess) {
        float[] weights = new float[numClass];
        if (!satisfyWeight(sumHess) || Maths.areZeros(sumGrad)) {
            return weights;
        }
        // TODO: regularization
        if (!fullHessian) {
            if (regAlpha == 0.0f) {
                for (int k = 0; k < numClass; k++)
                    weights[k] = -sumGrad[k] / (sumHess[k] + regLambda);
            } else {
                for (int k = 0; k < numClass; k++)
                    weights[k] = -Maths.thresholdL1(sumGrad[k], regAlpha) / (sumHess[k] + regLambda);
            }
        } else {
            addDiagonal(numClass, sumHess, regLambda);
            weights = Maths.solveLinearSystemWithCholeskyDecomposition(sumHess, sumGrad, numClass);
            for (int i = 0; i < numClass; i++)
                weights[i] *= -1;
            addDiagonal(numClass, sumHess, -regLambda);
        }
        return weights;
    }

    /**
     * Calculate the cost of loss function
     *
     * @param sumGrad sum of gradient values
     * @param sumHess sum of hessian values
     * @return loss gain
     */
    public float calcGain(float sumGrad, float sumHess) {
        if (!satisfyWeight(sumHess) || sumGrad == 0.0f) {
            return 0.0f;
        }
        if (maxLeafWeight == 0.0f) {
            if (regAlpha == 0.0f) {
                return (sumGrad / (sumHess + regLambda)) * sumGrad;
            } else {
                return Maths.sqr(Maths.thresholdL1(sumGrad, regAlpha)) / (sumHess + regLambda);
            }
        } else {
            float w = calcWeight(sumGrad, sumHess);
            float ret = sumGrad * w + 0.5f * (sumHess + regLambda) * Maths.sqr(w);
            if (regAlpha == 0.0f) {
                return -2.0f * ret;
            } else {
                return -2.0f * (ret + regAlpha * Math.abs(w));
            }
        }
    }

    public float calcGain(float[] sumGrad, float[] sumHess) {
        double gain = 0.0;
        if (!satisfyWeight(sumHess) || Maths.areZeros(sumGrad)) {
            return 0.0f;
        }
        // TODO: regularization
        if (!fullHessian) {
            if (regAlpha == 0.0f) {
                for (int k = 0; k < numClass; k++)
                    gain += sumGrad[k] / (sumHess[k] + regLambda) * sumGrad[k];
            } else {
                for (int k = 0; k < numClass; k++)
                    gain += Maths.sqr(Maths.thresholdL1(sumGrad[k], regAlpha)) * (sumHess[k] + regLambda);
            }
        } else {
            addDiagonal(numClass, sumHess, regLambda);
            float[] tmp = Maths.solveLinearSystemWithCholeskyDecomposition(sumHess, sumGrad, numClass);
            gain = Maths.dot(sumGrad, tmp);
            addDiagonal(numClass, sumHess, -regLambda);
        }
        return (float) (gain / numClass);
    }

    private void addDiagonal(int n, float[] sumHess, float v) {
        for (int i = 0; i < n; i++) {
            int index = Maths.indexOfLowerTriangularMatrix(i, i);
            sumHess[index] += v;
        }
    }

}

