package com.henrygouk.sgt;

import java.io.Serializable;
import java.util.Random;

public class StreamingGradientTree implements Serializable {

    private static final long serialVersionUID = -6696866670542821458L;

    protected FeatureInfo[] mFeatureInfo;

    protected StreamingGradientTreeOptions mOptions;

    protected Node mRoot;

    protected int mNumNodes;

    protected int mNumNodeUpdates;

    protected int mNumSplits;

    protected int mMaxDepth;
    
    public StreamingGradientTree(FeatureInfo[] featureInfo, StreamingGradientTreeOptions options) {
        mFeatureInfo = featureInfo.clone();
        mOptions = options;
        boolean[] hasSplit = new boolean[mFeatureInfo.length];

        for(int i = 0; i < hasSplit.length; i++) {
            hasSplit[i] = false;
        }

        mRoot = new Node(options.initialPrediction, 1, hasSplit);
    }

    public int getNumNodes() {
        return mNumNodes;
    }

    public int getNumNodeUpdates() {
        return mNumNodeUpdates;
    }

    public int getNumSplits() {
        return mNumSplits;
    }

    public int getDepth() {
        return mMaxDepth;
    }

    public void randomlyInitialize(Random rng, double predBound) {
        int fid = rng.nextInt(mFeatureInfo.length);
        mRoot.mSplit = new Split();
        mRoot.mSplit.feature = fid;
        mNumSplits++;

        boolean[] hasSplit = new boolean[mFeatureInfo.length];

        for(int i = 0; i < hasSplit.length; i++) {
            hasSplit[i] = false;
        }

        // If nominal
        if(mFeatureInfo[fid].type == FeatureType.nominal) {
            mRoot.mChildren = new Node[mFeatureInfo[fid].categories];

            for(int i = 0; i < mRoot.mChildren.length; i++) {
                mRoot.mChildren[i] = new Node(predBound * (2.0 * rng.nextDouble() - 1.0), 2, hasSplit);
            }
        }
        else if(mFeatureInfo[fid].type == FeatureType.ordinal) {
            mRoot.mSplit.index = rng.nextInt(mFeatureInfo[fid].categories / 2) + mFeatureInfo[fid].categories / 4;

            mRoot.mChildren = new Node[2];
            mRoot.mChildren[0] = new Node(predBound * (2.0 * rng.nextDouble() - 1.0), 2, hasSplit);
            mRoot.mChildren[1] = new Node(predBound * (2.0 * rng.nextDouble() - 1.0), 2, hasSplit);
        }
    }

    public void update(int[] features, GradHess gradHess) {
        Node leaf = mRoot.getLeaf(features);
        leaf.update(features, gradHess);

        if(leaf.mInstances % mOptions.gracePeriod != 0) {
            return;
        }

        Split bestSplit = leaf.findBestSplit();

        double p = computePValue(bestSplit, leaf.mInstances);

        if(p < mOptions.delta && bestSplit.lossMean < 0.0) {
            leaf.applySplit(bestSplit);
        }
    }

    public double predict(int[] features) {
        return mRoot.getLeaf(features).predict();
    }

    protected double computePValue(Split split, int instances) {
        // H0: the expected loss is zero
        // HA: the expected loss is not zero

        try {
            double F = instances * Math.pow(split.lossMean, 2.0) / split.lossVariance;

            return Statistics.FProbability(F, 1, instances - 1);
        }
        catch(ArithmeticException e) {
            System.err.println(e.getMessage());
            System.err.println(split.lossMean + " " + split.lossVariance);

            return 1.0;
        }
    }

    protected class Split implements Serializable {
        private static final long serialVersionUID = -6153673818687099581L;

        //lossMean and lossVariance are actually statistics of the approximation to the *change* in loss.

        double lossMean = 0;
        double lossVariance = 0;
        double[] deltaPredictions;
        int feature = -1;
        int index = -1;
    }

    protected class Node implements Serializable {

        private static final long serialVersionUID = -3259526711954744617L;

        protected double mPrediction;

        protected Node[] mChildren;

        protected Split mSplit;

        protected int mInstances;

        protected GradHessStats mUpdateStats;

        protected GradHessStats[][] mSplitStats;

        protected int mDepth;

        protected boolean[] mHasSplit;
        
        public Node(double prediction, int depth, boolean[] hasSplit) {
            mPrediction = prediction;
            mNumNodes++;
            mDepth = depth;
            mMaxDepth = Math.max(mMaxDepth, mDepth);
            mHasSplit = hasSplit.clone();

            reset();
        }

        public void reset() {
            mSplitStats = new GradHessStats[mFeatureInfo.length][];
            mUpdateStats = new GradHessStats();
            mInstances = 0;

            for(int i = 0; i < mSplitStats.length; i++) {
                mSplitStats[i] = new GradHessStats[mFeatureInfo[i].categories];

                for(int j = 0; j < mSplitStats[i].length; j++) {
                    mSplitStats[i][j] = new GradHessStats();
                }
            }
        }

        public Node getLeaf(int[] features) {
            if(mChildren == null) {
                return this;
            }
            else {
                FeatureType featureType = mFeatureInfo[mSplit.feature].type;
                Node c = null;
                
                if(features[mSplit.feature] == -1) {
                    c = mChildren[0];
                }
                else if(featureType == FeatureType.nominal) {
                    c = mChildren[features[mSplit.feature]];
                }
                else if(featureType == FeatureType.ordinal) {
                    if(features[mSplit.feature] <= mSplit.index) {
                        c = mChildren[0];
                    }
                    else {
                        c = mChildren[1];
                    }
                }
                else {
                    System.err.println("Unhandled attribute type");
                }

                return c.getLeaf(features);
            }
        }

        void update(int[] features, GradHess gradHess) {
            mInstances++;

            for(int i = 0; i < features.length; i++) {
                if(features[i] == -1) {
                    continue;
                }

                mSplitStats[i][features[i]].addObservation(gradHess);
            }

            mUpdateStats.addObservation(gradHess);
        }

        public double predict() {
            return mPrediction;
        }

        public Split findBestSplit() {
            
            Split best = new Split();

            // We can try to update the prediction using the new gradient information
            best.deltaPredictions = new double[] {computeDeltaPrediction(mUpdateStats.getMean())};
            best.lossMean = mUpdateStats.getDeltaLossMean(best.deltaPredictions[0]);
            best.lossVariance = mUpdateStats.getDeltaLossVariance(best.deltaPredictions[0]);
            best.feature = -1;
            best.index = -1;

            for(int i = 0; i < mSplitStats.length; i++) {
                Split candidate = new Split();
                candidate.feature = i;

                if(mFeatureInfo[i].type == FeatureType.nominal) {
                    if(mHasSplit[i]) {
                        continue;
                    }

                    candidate.deltaPredictions = new double[mSplitStats[i].length];
                    double lossMean = 0.0;
                    double lossVar = 0.0;
                    int observations = 0;

                    for(int j = 0; j < mSplitStats[i].length; j++) {
                        double p = computeDeltaPrediction(mSplitStats[i][j].getMean());
                        double m = mSplitStats[i][j].getDeltaLossMean(p);
                        double s = mSplitStats[i][j].getDeltaLossVariance(p);
                        int n = mSplitStats[i][j].getObservationCount();
                        candidate.deltaPredictions[j] = p;

                        lossMean = GradHessStats.combineMean(lossMean, observations, m, n);
                        lossVar = GradHessStats.combineVariance(lossMean, lossVar, observations, m, s, n);
                        observations += n;
                    }

                    candidate.lossMean = lossMean + mSplitStats[i].length * mOptions.gamma / mInstances;
                    candidate.lossVariance = lossVar;
                }
                else if(mFeatureInfo[i].type == FeatureType.ordinal) {
                    GradHessStats[] forwardCumulativeSum = new GradHessStats[mFeatureInfo[i].categories - 1];
                    GradHessStats[] backwardCumulativeSum = new GradHessStats[mFeatureInfo[i].categories - 1];

                    // Compute the split stats for each possible split point
                    for(int j = 0; j < mFeatureInfo[i].categories - 1; j++) {
                        forwardCumulativeSum[j] = new GradHessStats();
                        forwardCumulativeSum[j].add(mSplitStats[i][j]);

                        if(j > 0) {
                            forwardCumulativeSum[j].add(forwardCumulativeSum[j - 1]);
                        }
                    }

                    for(int j = mFeatureInfo[i].categories - 2; j >= 0; j--) {
                        backwardCumulativeSum[j] = new GradHessStats();
                        backwardCumulativeSum[j].add(mSplitStats[i][j + 1]);

                        if(j + 1 < backwardCumulativeSum.length) {
                            backwardCumulativeSum[j].add(backwardCumulativeSum[j + 1]);
                        }
                    }

                    candidate.lossMean = Double.POSITIVE_INFINITY;
                    candidate.deltaPredictions = new double[2];

                    for(int j = 0; j < forwardCumulativeSum.length; j++) {
                        double deltaPredLeft = computeDeltaPrediction(forwardCumulativeSum[j].getMean());
                        double lossMeanLeft = forwardCumulativeSum[j].getDeltaLossMean(deltaPredLeft);
                        double lossVarLeft = forwardCumulativeSum[j].getDeltaLossVariance(deltaPredLeft);
                        int numLeft = forwardCumulativeSum[j].getObservationCount();

                        double deltaPredRight = computeDeltaPrediction(backwardCumulativeSum[j].getMean());
                        double lossMeanRight = backwardCumulativeSum[j].getDeltaLossMean(deltaPredRight);
                        double lossVarRight = backwardCumulativeSum[j].getDeltaLossVariance(deltaPredRight);
                        int numRight = backwardCumulativeSum[j].getObservationCount();

                        double lossMean = GradHessStats.combineMean(lossMeanLeft, numLeft, lossMeanRight, numRight);
                        double lossVar = GradHessStats.combineVariance(lossMeanLeft, lossVarLeft, numLeft, lossMeanRight, lossVarRight, numRight);

                        if(lossMean < candidate.lossMean) {
                            candidate.lossMean = lossMean + 2.0 * mOptions.gamma / mInstances;
                            candidate.lossVariance = lossVar;
                            candidate.index = j;
                            candidate.deltaPredictions[0] = deltaPredLeft;
                            candidate.deltaPredictions[1] = deltaPredRight;
                        }
                    }
                }
                else {
                    System.err.println("Unhandled attribute type");
                }

                if(candidate.lossMean < best.lossMean) {
                    best = candidate;
                }
            }
            
            return best;
        }

        public void applySplit(Split split) {

            //Should we just update the prediction being made?
            if(split.feature == -1) {
                mPrediction += split.deltaPredictions[0];
                mNumNodeUpdates++;
                reset();
                return;
            }

            mSplit = split;
            mNumSplits++;
            mHasSplit[split.feature] = true;
            
            if(mFeatureInfo[split.feature].type == FeatureType.nominal) {
                mChildren = new Node[mFeatureInfo[split.feature].categories];

                for(int i = 0; i < mChildren.length; i++) {
                    mChildren[i] = new Node(mPrediction + split.deltaPredictions[i], mDepth + 1, mHasSplit);
                }
            }
            else if(mFeatureInfo[split.feature].type == FeatureType.ordinal) {
                mChildren = new Node[2];

                mChildren[0] = new Node(mPrediction + split.deltaPredictions[0], mDepth + 1, mHasSplit);
                mChildren[1] = new Node(mPrediction + split.deltaPredictions[1], mDepth + 1, mHasSplit);
            }
            else {
                System.err.println("Unhandled attribute type");
            }

            //Free up memory used by the split stats
            mSplitStats = null;
        }

        protected double computeDeltaPrediction(GradHess gradHess) {
            return -gradHess.gradient / (gradHess.hessian + Double.MIN_NORMAL + mOptions.lambda);
        }
    }
}