package com.henrygouk.sgt;

import java.io.Serializable;
import java.util.Random;
import java.util.stream.IntStream;

public class StreamingGradientTreeCommittee implements Serializable {

    protected StreamingGradientTree[] mTrees;

    public StreamingGradientTreeCommittee(FeatureInfo[] featureInfo, StreamingGradientTreeOptions options, int numTrees) {
        mTrees = new StreamingGradientTree[numTrees];
        
        for(int i = 0; i < mTrees.length; i++) {
            mTrees[i] = new StreamingGradientTree(featureInfo, options);
        }
    }

    public int getNumNodes() {
        int result = 0;

        for(int i = 0; i < mTrees.length; i++) {
            result += mTrees[i].getNumNodes();
        }

        return result;
    }

    public int getNumNodeUpdates() {
        int result = 0;

        for(int i = 0; i < mTrees.length; i++) {
            result += mTrees[i].getNumNodeUpdates();
        }

        return result;
    }

    public int getNumSplits() {
        int result = 0;

        for(int i = 0; i < mTrees.length; i++) {
            result += mTrees[i].getNumSplits();
        }

        return result;
    }

    public int getMaxDepth() {
        int result = 0;

        for(int i = 0; i < mTrees.length; i++) {
            result = Math.max(mTrees[i].getDepth(), result);
        }

        return result;
    }

    public int getNumTrees() {
        return mTrees.length;
    }

    public void randomlyInitialize(Random rng, double predBound) {
        for(StreamingGradientTree t : mTrees) {
            t.randomlyInitialize(rng, predBound);
        }
    }

    public void update(int[] features, GradHess[] gradHesses) {
        IntStream.range(0, mTrees.length)
                 .parallel()
                 .forEach(i -> mTrees[i].update(features, gradHesses[i]));
    }

    public double[] predict(int[] features) {
        return IntStream.range(0, mTrees.length)
                        .parallel()
                        .mapToDouble(i -> mTrees[i].predict(features))
                        .toArray();
    }
}