package com.henrygouk.sgt.neural;

import java.io.Serializable;

import com.henrygouk.sgt.GradHess;

public class RectifiedLinearUnit implements Serializable, Layer {

    private static final long serialVersionUID = 1L;

    public GradHess[] update(double[] features, GradHess[] gradHess) {
        GradHess[] result = new GradHess[features.length];

        for(int i = 0; i < result.length; i++) {
            if(features[i] > 0.0) {
                result[i] = new GradHess(gradHess[i]);
            }
            else {
                result[i] = new GradHess();
            }
        }

        return result;
    }

    public double[] predict(double[] features) {
        double[] result = new double[features.length];

        for(int i = 0; i < result.length; i++) {
            result[i] = features[i] > 0.0 ? features[i] : 0.0;
        }

        return result;
    }
}