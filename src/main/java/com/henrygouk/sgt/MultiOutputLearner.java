package com.henrygouk.sgt;

import java.util.Random;

public interface MultiOutputLearner {

    public int getNumNodes();

    public int getNumNodeUpdates();

    public int getNumSplits();

    public int getMaxDepth();

    public int getNumTrees();

    public void randomlyInitialize(Random rng, double predBound);

    public void update(int[] features, GradHess[] gradHess);

    public double[] predict(int[] features);
}