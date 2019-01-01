package com.henrygouk.sgt;

import static org.junit.Assert.assertEquals;

import org.junit.*;

public class GradHessStatsTest {
    @Test
    public void testGetDeltaLossMean() {
        GradHessStats stats = new GradHessStats();
        stats.addObservation(new GradHess(0.5, 1.0));
        stats.addObservation(new GradHess(1.0, 1.0));
        stats.addObservation(new GradHess(1.5, 1.0));
        double deltaPrediction = -1.0;
        double deltaLossMean = stats.getDeltaLossMean(deltaPrediction);
        
        assertEquals(-0.5, deltaLossMean, 1E-12);
    }

    @Test
    public void testGetDeltaLossVariance() {
        GradHessStats stats = new GradHessStats();
        stats.addObservation(new GradHess(0.5, 1.0));
        stats.addObservation(new GradHess(1.0, 1.0));
        stats.addObservation(new GradHess(1.5, 1.0));
        double deltaPrediction = -1.0;
        double deltaLossVariance = stats.getDeltaLossVariance(deltaPrediction);

        assertEquals(0.25, deltaLossVariance, 1E-12);
    }

    @Test
    public void testAdd() {
        GradHessStats stats1 = new GradHessStats();
        stats1.addObservation(new GradHess(0.5, 0.8));
        stats1.addObservation(new GradHess(1.0, 1.0));
        stats1.addObservation(new GradHess(1.5, 1.2));

        GradHessStats stats2 = new GradHessStats();
        stats2.addObservation(new GradHess(0.5, 1.2));
        stats2.addObservation(new GradHess(1.0, 1.0));
        stats2.addObservation(new GradHess(1.5, 0.9));

        stats1.add(stats2);

        GradHessStats expected = new GradHessStats();
        expected.addObservation(new GradHess(0.5, 0.8));
        expected.addObservation(new GradHess(1.0, 1.0));
        expected.addObservation(new GradHess(1.5, 1.2));
        expected.addObservation(new GradHess(0.5, 1.2));
        expected.addObservation(new GradHess(1.0, 1.0));
        expected.addObservation(new GradHess(1.5, 0.9));

        assertEquals(expected.getVariance().gradient, stats1.getVariance().gradient, 1E-12);
        assertEquals(expected.getCovariance(), stats1.getCovariance(), 1E-12);
    }

    @Test
    public void testCombineMean() {
        double m1 = 2.0;
        int n1 = 5;
        double m2 = 3.0;
        int n2 = 3;
        double m = 19.0 / 8.0;

        assertEquals(m, GradHessStats.combineMean(m1, n1, m2, n2), 1E-12);
    }

    public void testCombineVariance() {
        double m1 = 2.0;
        double s1 = 1.0;
        int n1 = 5;
        double m2 = 3.0;
        double s2 = 4.0;
        int n2 = 3;
        double m = 19.0 / 8.0;

        assertEquals(1.9821, GradHessStats.combineVariance(m1, s1, n1, m2, s2, n2), 1E-12);
    }
}