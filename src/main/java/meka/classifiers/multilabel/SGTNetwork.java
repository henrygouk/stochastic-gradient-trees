package meka.classifiers.multilabel;

import java.util.Random;

import com.henrygouk.sgt.*;
import com.henrygouk.sgt.neural.*;

import meka.classifiers.multilabel.AbstractMultiLabelClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class SGTNetwork extends AbstractMultiLabelClassifier {

    private static final long serialVersionUID = 8498761704450406114L;

    protected int mBins = 64;
    protected StreamingGradientTreeNetwork mNetwork;
    protected int mNumTrees = 100;
    protected int mNumHidden = 100;
    protected int mBatchSize = 100;
    protected double mLearningRate = 0.001;
    protected double mBeta1 = 0.9;
    protected double mBeta2 = 0.999;
    protected double mEpsilon = 1E-8;
    protected int mEpochs = 20;
    protected Objective mObjective;
    protected Discretize mDiscretize;

    public int getEpochs() {
        return mEpochs;
    }

    public void setEpochs(int epochs) {
        mEpochs = epochs;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        FeatureInfo[] featureInfo = createFeatureInfo(data);

        Layer[] layers = new Layer[] {
            new FullyConnected(mNumTrees, mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumHidden, mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumHidden, data.classIndex(), mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon)
        };

        StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
        options.gracePeriod = mBatchSize;
        options.lambda = 0.1;
        options.gamma = 1.0;

        mNetwork = new StreamingGradientTreeNetwork(featureInfo, options, mNumTrees, layers);
        mNetwork.randomlyInitialize(new Random(), 1.0 / Math.sqrt(mNumTrees));
        mObjective = new BinaryCrossEntropy();

        for(int e = 0; e < mEpochs; e++) {
            for(int i = 0; i < data.numInstances(); i++) {
                int[] features = getFeatures(data.instance(i));

                double[] pred = mNetwork.predict(features);
                double[] groundTruth = new double[data.classIndex()];
                groundTruth = new double[data.classIndex()];

                for(int l = 0; l < data.classIndex(); l++) {
                    groundTruth[l] = data.instance(i).value(l);
                }

                GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, pred);

                mNetwork.update(features, gradHess);
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance inst) {
        int[] features = getFeatures(inst);
        double[] pred = mNetwork.predict(features);

        return mObjective.transfer(pred);
    }
    
    private FeatureInfo[] createFeatureInfo(Instances insts) throws Exception {
        FeatureInfo[] featureInfo = new FeatureInfo[insts.numAttributes() - insts.classIndex()];
        int i = 0;

        for(int attInd = 0; attInd < insts.numAttributes(); attInd++) {
            if(attInd < insts.classIndex()) {
                continue;
            }

            featureInfo[i] = new FeatureInfo();
            
            if(insts.attribute(attInd).isNominal()) {
                featureInfo[i].type = FeatureType.nominal;
                featureInfo[i].categories = insts.attribute(attInd).numValues();
            }
            else if(insts.attribute(attInd).isNumeric()) {
                featureInfo[i].type = FeatureType.ordinal;
                featureInfo[i].categories = mBins;
            }

            i++;
        }

        mDiscretize = new Discretize();
        mDiscretize.setUseEqualFrequency(true);
        mDiscretize.setBins(mBins);
        mDiscretize.setInputFormat(insts);
        Filter.useFilter(insts, mDiscretize);

        return featureInfo;
    }

    private int[] getFeatures(Instance inst) {
        mDiscretize.input(inst);
        mDiscretize.batchFinished();
        inst = mDiscretize.output();
        
        int[] features = new int[inst.numAttributes() - inst.classIndex()];

        int i = 0;

        for(int attInd = 0; attInd < inst.numAttributes(); attInd++) {
            if(attInd < inst.classIndex()) {
                continue;
            }

            if(Double.isNaN(inst.value(attInd))) {
                features[i] = -1;
            }
            else if(inst.attribute(attInd).isNominal()) {
                features[i] = (int)inst.value(attInd);
            }

            i++;
        }

        return features;
    }

}