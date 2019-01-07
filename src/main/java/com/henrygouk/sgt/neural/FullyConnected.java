package com.henrygouk.sgt.neural;

import java.io.Serializable;
import java.util.Random;

import com.henrygouk.sgt.GradHess;

public class FullyConnected implements Serializable, Layer {
    
    private static final long serialVersionUID = 1L;

    protected double[] mWeights;

    protected double[] mBiases;

    protected double[] mWeightGrads;

    protected double[] mBiasGrads;

    protected double[] mWeightMomentum;

    protected double[] mBiasMomentum;

    protected double[] mWeightVelocity;

    protected double[] mBiasVelocity;

    protected int mUpdates;

    protected int mSteps;

    protected int mInputs;
    
    protected int mOutputs;

    protected int mBatchSize;

    protected double mLearningRate;

    protected double mBeta1;
    
    protected double mBeta2;

    protected double mEpsilon;

    public double getLearningRate() {
        return mLearningRate;
    }

    public void setLearningRate(double learningRate) {
        mLearningRate = learningRate;
    }

    public FullyConnected(int numInputs, int numOutputs, int batchSize, double learningRate, double beta1, double beta2, double epsilon) {
        mInputs = numInputs;
        mOutputs = numOutputs;
        mBatchSize = batchSize;
        mLearningRate = learningRate;
        mBeta1 = beta1;
        mBeta2 = beta2;
        mEpsilon = epsilon;
        mWeights = new double[numInputs * numOutputs];
        mWeightGrads = new double[numInputs * numOutputs];
        mWeightMomentum = new double[numInputs * numOutputs];
        mWeightVelocity = new double[numInputs * numOutputs];
        mBiases = new double[mOutputs];
        mBiasGrads = new double[mOutputs];
        mBiasMomentum = new double[mOutputs];
        mBiasVelocity = new double[mOutputs];

        Random rng = new Random(1);

        //Xavier Glorot initialisation!
        double stddev = Math.sqrt(2.0 / (mInputs + mOutputs));

        for(int i = 0; i < mWeights.length; i++) {
            mWeights[i] = rng.nextGaussian() * stddev;
        }
    }

    public GradHess[] update(double[] features, GradHess[] gradHess) {
        GradHess[] backProp = new GradHess[mInputs];

        for(int i = 0; i < mInputs; i++) {
            backProp[i] = new GradHess();
        }

        for(int o = 0; o < mOutputs; o++) {
            for(int i = 0; i < mInputs; i++) {
                // Compute backprop term that depends on this weight
                backProp[i].gradient += mWeights[o * mInputs + i] * gradHess[o].gradient;
                backProp[i].hessian += Math.pow(mWeights[o * mInputs + i], 2.0) * gradHess[o].hessian;

                // Compute values needed for updating this layer
                mWeightGrads[o * mInputs + i] += gradHess[o].gradient * features[i] / mBatchSize;
            }

            mBiasGrads[o] += gradHess[o].gradient / mBatchSize;
        }

        mUpdates++;

        if(mUpdates == mBatchSize) {
            mSteps++;
            mUpdates = 0;
            int j = 0;

            for(int i = 0; i < mInputs; i++) {
                for(int o = 0; o < mOutputs; o++, j++) {
                    mWeightMomentum[j] = mBeta1 * mWeightMomentum[j] + (1.0 - mBeta1) * mWeightGrads[j];
                    mWeightVelocity[j] = mBeta2 * mWeightVelocity[j] + (1.0 - mBeta2) * Math.pow(mWeightGrads[j], 2.0);
                    double m = mWeightMomentum[j] / (1.0 - Math.pow(mBeta1, mSteps));
                    double v = mWeightVelocity[j] / (1.0 - Math.pow(mBeta2, mSteps));
                    mWeights[j] -= mLearningRate * m / (Math.sqrt(v) + mEpsilon);

                    mWeightGrads[j] = 0.0;
                }
            }

            for(int o = 0; o < mOutputs; o++) {
                mBiasMomentum[o] = mBeta1 * mBiasMomentum[o] + (1.0 - mBeta1) * mBiasGrads[o];
                mBiasVelocity[o] = mBeta2 * mBiasVelocity[o] + (1.0 - mBeta2) * Math.pow(mBiasGrads[o], 2.0);
                double m = mBiasMomentum[o] / (1.0 - Math.pow(mBeta1, mSteps));
                double v = mBiasVelocity[o] / (1.0 - Math.pow(mBeta2, mSteps));
                mBiases[o] -= mLearningRate * m / (Math.sqrt(v) + mEpsilon);

                mBiasGrads[o] = 0.0;
            }
        }

        return backProp;
    }

    public double[] predict(double[] features) {
        double[] result = mBiases.clone();

        for(int o = 0; o < result.length; o++) {
            for(int i = 0; i < features.length; i++) {
                result[o] += mWeights[o * mInputs + i] * features[i];
            }
        }

        return result;
    }
}