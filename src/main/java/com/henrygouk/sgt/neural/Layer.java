package com.henrygouk.sgt.neural;

import com.henrygouk.sgt.GradHess;

public interface Layer {
    public GradHess[] update(double[] features, GradHess[] gradHess);

    public double[] predict(double[] features);
}