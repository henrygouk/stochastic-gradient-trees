package com.henrygouk.sgt;

public abstract class Objective {
    public abstract GradHess[] computeDerivatives(double[] groundTruth, double[] raw);

    public double[] transfer(double[] raw) {
        return raw;
    }
}