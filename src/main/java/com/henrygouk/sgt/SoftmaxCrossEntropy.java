package com.henrygouk.sgt;

import java.io.Serializable;

public class SoftmaxCrossEntropy extends Objective implements Serializable {

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
        double[] result = new double[raw.length + 1];

        for(int i = 0; i < raw.length; i++) {
            result[i] = raw[i];
        }

        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;

        for(int i = 0; i < result.length; i++) {
            max = Math.max(max, result[i]);
        }

        for(int i = 0; i < result.length; i++) {
            result[i] = Math.exp(result[i] - max);
            sum += result[i];
        }

        for(int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

}