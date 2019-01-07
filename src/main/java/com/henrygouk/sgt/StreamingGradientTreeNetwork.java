package com.henrygouk.sgt;

import java.io.Serializable;
import java.util.Random;
import java.util.stream.IntStream;

import com.henrygouk.sgt.neural.Layer;

public class StreamingGradientTreeNetwork implements Serializable, MultiOutputLearner {

    private static final long serialVersionUID = -512245313461494205L;
    protected StreamingGradientTree[] mTrees;
    protected Layer[] mLayers;

    public StreamingGradientTreeNetwork(FeatureInfo[] featureInfo, StreamingGradientTreeOptions options, int numTrees, Layer[] layers) {
        mTrees = new StreamingGradientTree[numTrees];
        mLayers = layers;
        
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
        double[][] activations = new double[mLayers.length + 1][];
        activations[0] = IntStream.range(0, mTrees.length)
                                  .parallel()
                                  .mapToDouble(i -> mTrees[i].predict(features))
                                  .toArray();
        
        for(int i = 0; i < mLayers.length; i++) {
            activations[i + 1] = mLayers[i].predict(activations[i]);
        }

        for(int i = mLayers.length - 1; i >= 0; i--) {
            gradHesses = mLayers[i].update(activations[i], gradHesses);
        }

        final GradHess[] finalGradHesses = gradHesses;

        IntStream.range(0, mTrees.length)
                 .parallel()
                 .forEach(i -> mTrees[i].update(features, finalGradHesses[i]));
    }

    public double[] predict(int[] features) {
        double[][] activations = new double[mLayers.length + 1][];
        activations[0] = IntStream.range(0, mTrees.length)
                                  .parallel()
                                  .mapToDouble(i -> mTrees[i].predict(features))
                                  .toArray();
        
        for(int i = 0; i < mLayers.length; i++) {
            activations[i + 1] = mLayers[i].predict(activations[i]);
        }

        return activations[activations.length - 1];
    }
}