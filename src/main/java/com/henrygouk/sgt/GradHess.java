package com.henrygouk.sgt;

import java.io.Serializable;

public class GradHess implements Serializable {

    private static final long serialVersionUID = 1L;

    public double gradient;
    public double hessian;

    public GradHess() {
        gradient = 0;
        hessian = 0;
    }

    public GradHess(GradHess gradHess) {
        gradient = gradHess.gradient;
        hessian = gradHess.hessian;
    }

    public GradHess(double grad, double hess) {
        gradient = grad;
        hessian = hess;
    }

    void add(GradHess gradHess) {
        gradient += gradHess.gradient;
        hessian += gradHess.hessian;
    }

    void sub(GradHess gradHess) {
        gradient -= gradHess.gradient;
        hessian -= gradHess.hessian;
    }
}