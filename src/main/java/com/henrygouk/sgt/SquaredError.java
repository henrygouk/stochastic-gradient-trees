package com.henrygouk.sgt;

import java.io.Serializable;

public class SquaredError extends Objective implements Serializable {

	private static final long serialVersionUID = 1L;

    @Override
	public GradHess[] computeDerivatives(double[] groundTruth, double[] raw) {
        GradHess[] result = new GradHess[raw.length];

        for(int i = 0; i < result.length; i++) {
            result[i] = new GradHess(raw[i] - groundTruth[i], 1.0);
        }

        return result;
	}

}