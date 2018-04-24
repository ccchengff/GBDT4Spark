package org.dma.gbdt4spark.algo.gbdt.histogram;

import org.dma.gbdt4spark.tree.param.GBDTParam;

import java.io.Serializable;

public class BinaryGradPair implements GradPair, Serializable {
    private float grad;
    private float hess;

    public BinaryGradPair() {}

    public BinaryGradPair(float grad, float hess) {
        this.grad = grad;
        this.hess = hess;
    }

    @Override
    public void plusBy(GradPair gradPair) {
        this.grad += ((BinaryGradPair) gradPair).grad;
        this.hess += ((BinaryGradPair) gradPair).hess;
    }

    @Override
    public void subtractBy(GradPair gradPair) {
        this.grad -= ((BinaryGradPair) gradPair).grad;
        this.hess -= ((BinaryGradPair) gradPair).hess;
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
        this.grad *= x;
        this.hess *= x;
    }

    @Override
    public float calcGain(GBDTParam param) {
        return param.calcGain(grad, hess);
    }

    public float calcWeight(GBDTParam param) {
        return param.calcWeight(grad, hess);
    }

    @Override
    public boolean satisfyWeight(GBDTParam param) {
        return param.satisfyWeight(hess);
    }

    @Override
    public GradPair copy() {
        return new BinaryGradPair(grad, hess);
    }

    public float getGrad() {
        return grad;
    }

    public float getHess() {
        return hess;
    }

    @Override
    public String toString() {
        return "(" + grad + ", " + hess + ")";
    }
}
