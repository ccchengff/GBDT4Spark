package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.tree.param.GBDTParam;

import java.io.Serializable;
import java.util.Arrays;

public class MultiGradPair implements GradPair, Serializable {
    private float[] grad;
    private float[] hess;

    public MultiGradPair(int numClass, boolean fullHessian) {
        this.grad = new float[numClass];
        if (fullHessian)
            this.hess = new float[(numClass * (numClass + 1)) >> 1];
        else
            this.hess = new float[numClass];
    }

    public MultiGradPair(float[] grad, float[] hess) {
        this.grad = grad;
        this.hess = hess;
    }

    @Override
    public void plusBy(GradPair gradPair) {
        float[] grad = ((MultiGradPair) gradPair).grad;
        float[] hess = ((MultiGradPair) gradPair).hess;
        for (int i = 0; i < this.grad.length; i++)
            this.grad[i] += grad[i];
        for (int i = 0; i < this.hess.length; i++)
            this.hess[i] += hess[i];
    }

    @Override
    public void subtractBy(GradPair gradPair) {
        float[] grad = ((MultiGradPair) gradPair).grad;
        float[] hess = ((MultiGradPair) gradPair).hess;
        for (int i = 0; i < this.grad.length; i++)
            this.grad[i] -= grad[i];
        for (int i = 0; i < this.hess.length; i++)
            this.hess[i] -= hess[i];
    }

    @Override
    public GradPair plus(GradPair gradPair) {
        GradPair res = this.copy();
        res.plusBy(gradPair);
        return res;
    }

    @Override
    public GradPair subtract(GradPair gradPair) {
        GradPair res = this.copy();
        res.subtractBy(gradPair);
        return res;
    }

    @Override
    public void timesBy(float x) {
        for (int i = 0; i < this.grad.length; i++)
            this.grad[i] *= x;
        for (int i = 0; i < this.hess.length; i++)
            this.hess[i] *= x;
    }

    @Override
    public float calcGain(GBDTParam param) {
        return param.calcGain(grad, hess);
    }

    public float[] calcWeights(GBDTParam param) {
        return param.calcWeights(grad, hess);
    }

    @Override
    public boolean satisfyWeight(GBDTParam param) {
        return param.satisfyWeight(hess);
    }

    @Override
    public MultiGradPair copy() {
        return new MultiGradPair(grad.clone(), hess.clone());
    }

    public float[] getGrad() {
        return grad;
    }

    public float[] getHess() {
        return hess;
    }

    @Override
    public String toString() {
        String gradStr = Arrays.toString(grad);
        if (grad.length == hess.length) {
            return "(" + gradStr + ", diag{" + Arrays.toString(hess) + "})";
        } else {
            int rowSize = 1, offset = 0;
            StringBuilder hessSB = new StringBuilder("[");
            while (rowSize <= grad.length) {
                hessSB.append("[");
                hessSB.append(hess[offset]);
                for (int i = 1; i < rowSize; i++) {
                    hessSB.append(", ");
                    hessSB.append(hess[offset + i]);
                }
                hessSB.append("]");
                offset += rowSize;
                rowSize++;
            }
            hessSB.append("]");
            return "(" + gradStr + ", " + hessSB.toString() + ")";
        }
    }
}
