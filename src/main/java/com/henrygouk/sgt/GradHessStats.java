package com.henrygouk.sgt;

import java.io.Serializable;

public class GradHessStats implements Serializable {

    private static final long serialVersionUID = 1L;

    protected GradHess mSum;
    
    protected GradHess mScaledVariance;

    protected double mScaledCovariance;

    protected int mObservations;

    public GradHessStats() {
        mSum = new GradHess();
        mScaledVariance = new GradHess();
        mScaledCovariance = 0.0;
    }

    public void add(GradHessStats stats) {
        if(stats.mObservations == 0) {
            return;
        }

        if(mObservations == 0) {
            mSum = new GradHess(stats.mSum);
            mScaledVariance = new GradHess(stats.mScaledVariance);
            mScaledCovariance = stats.mScaledCovariance;
            mObservations = stats.mObservations;

            return;
        }

        GradHess meanDiff = stats.getMean();
        meanDiff.sub(getMean());
        int n1 = mObservations;
        int n2 = stats.mObservations;

        // Do scaled variance bit (see Wikipedia page on "Algorithms for calculating variance", section about parallel calculation)
        mScaledVariance.gradient += stats.mScaledVariance.gradient + Math.pow(meanDiff.gradient, 2.0) * (n1 * n2) / (n1 + n2);
        mScaledVariance.hessian += stats.mScaledVariance.hessian + Math.pow(meanDiff.hessian, 2.0) * (n1 * n2) / (n1 + n2);

        // Do scaled covariance bit (see "Numerically Stable, Single-Pass, Parallel Statistics Algorithms" (Bennett et al, 2009))
        mScaledCovariance += stats.mScaledCovariance + meanDiff.gradient * meanDiff.hessian * (n1 * n2) / (n1 + n2);

        // Do the other bits
        mSum.add(stats.mSum);
        mObservations += stats.mObservations;
    }

    public void addObservation(GradHess gradHess) {
        GradHess oldMean = getMean();
        mSum.add(gradHess);
        mObservations++;
        GradHess newMean = getMean();

        mScaledVariance.gradient += (gradHess.gradient - oldMean.gradient) * (gradHess.gradient - newMean.gradient);
        mScaledVariance.hessian += (gradHess.hessian - oldMean.hessian) * (gradHess.hessian - newMean.hessian);

        mScaledCovariance += (gradHess.gradient - oldMean.gradient) * (gradHess.hessian - newMean.hessian);
    }

    public GradHess getMean() {
        if(mObservations == 0) {
            return new GradHess(0.0, 0.0);
        }
        else {
            return new GradHess(mSum.gradient / mObservations, mSum.hessian / mObservations);
        }
    }

    public GradHess getVariance() {
        if(mObservations < 2) {
            return new GradHess(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
        else {
            return new GradHess(
                mScaledVariance.gradient / (mObservations - 1),
                mScaledVariance.hessian / (mObservations - 1)
            );
        }
    }

    public double getCovariance() {
        if(mObservations < 2) {
            return Double.POSITIVE_INFINITY;
        }
        else {
            return mScaledCovariance / (mObservations - 1);
        }
    }

    public int getObservationCount() {
        return mObservations;
    }

    public double getDeltaLossMean(double deltaPrediction) {
        GradHess mean = getMean();

        return deltaPrediction * mean.gradient + 0.5 * mean.hessian * Math.pow(deltaPrediction, 2.0);
    }

    /*
        This method ignores correlations between deltaPrediction and the gradients/hessians! Considering
        deltaPredicions is derived from the gradient and hessian sample, this assumption is definitely violated.
    */
    public double getDeltaLossVariance(double deltaPrediction) {
        GradHess variance = getVariance();
        double covariance = getCovariance();

        double gradTermVariance = Math.pow(deltaPrediction, 2.0) * variance.gradient;
        double hessTermVariance = 0.25 * variance.hessian * Math.pow(deltaPrediction, 4.0);

        return Math.max(0.0, gradTermVariance + hessTermVariance + Math.pow(deltaPrediction, 3.0) * covariance);
    }

    public static double combineMean(double m1, int n1, double m2, int n2) {
        if(n1 == 0) {
            return m2;
        }

        if(n2 == 0) {
            return m1;
        }
        
        return (m1 * n1 + m2 * n2) / (n1 + n2);
    }

    // This method accepts, and returns, unbiased estimates of the variance.
    public static double combineVariance(double m1, double s1, int n1, double m2, double s2, int n2) {
        // Some special cases, just to be safe
        if(n1 == 0) {
            return s2;
        }
        
        if(n2 == 0) {
            return s1;
        }

        double n = n1 + n2;
        double m = GradHessStats.combineMean(m1, n1, m2, n2);

        // First we have to bias the sample variances (we'll unbias this later)
        s1 = ((double)(n1 - 1) / n1) * s1;
        s2 = ((double)(n2 - 1) / n2) * s2;

        // Compute the sum of squares of all the datapoints
        double t1 = n1 * (s1 + m1 * m1);
        double t2 = n2 * (s2 + m2 * m2);
        double t = t1 + t2;

        // Now get the full (biased) sample variance
        double s = t / n - m;

        // Apply Bessel's correction
        s = ((double)n / (n - 1)) * s;

        return s;
    }
}