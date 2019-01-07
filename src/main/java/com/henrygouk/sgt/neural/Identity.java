package com.henrygouk.sgt.neural;

import java.io.Serializable;

import com.henrygouk.sgt.GradHess;

public class Identity implements Serializable, Layer {
    private static final long serialVersionUID = 1583592862930132577L;

    public GradHess[] update(double[] features, GradHess[] gradHess) {
        return gradHess.clone();
    }

    public double[] predict(double[] features) {
        return features.clone();
    }
}