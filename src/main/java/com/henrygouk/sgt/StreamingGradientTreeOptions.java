package com.henrygouk.sgt;

import java.io.Serializable;

public class StreamingGradientTreeOptions implements Serializable
{
    private static final long serialVersionUID = 8851226957419993860L;
    public double delta = 1E-7;
    public int gracePeriod = 200;
    public double initialPrediction = 0;
    public double lambda = 0.1;
    public double gamma = 1.0;
}