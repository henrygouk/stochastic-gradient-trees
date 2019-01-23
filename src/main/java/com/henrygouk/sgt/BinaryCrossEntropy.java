package com.henrygouk.sgt;

import java.io.Serializable;

public class BinaryCrossEntropy extends Objective implements Serializable {

    private static final long serialVersionUID = 1L;

    @Override
    public GradHess[] computeDerivatives(double[] groundTruth, double[] raw) {
        GradHess[] result = new GradHess[raw.length];
        double[] predictions = transfer(raw);

        for(int i = 0; i < result.length; i++) {
            result[i] = new GradHess(predictions[i] - groundTruth[i], predictions[i] * (1.0 - predictions[i]));
        }

        return result;
    }
    
    @Override
    public double[] transfer(double[] raw) {
        double[] result = new double[raw.length];

        for(int i = 0; i < result.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-raw[i]));
        }

        return result;
    }

}