package weka.classifiers.trees;

import java.util.Random;

import com.henrygouk.sgt.*;
import com.henrygouk.sgt.neural.*;

import scala.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class SGTNetwork extends AbstractClassifier {

    private static final long serialVersionUID = 3601534653937503693L;
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
    protected double mLambda = 0.1;
    protected double mGamma = 1.0;
    protected Stats mTargetStats;

    public int getEpochs() {
        return mEpochs;
    }

    public void setEpochs(int epochs) {
        mEpochs = epochs;
    }

    public void setBins(int bins) {
        mBins = bins;
    }

    public int getBins() {
        return mBins;
    }

    public void setTrainBatchSize(int bs) {
        mBatchSize = bs;
    }

    public int getTrainBatchSize() {
        return mBatchSize;
    }

    public void setLambda(double l) {
        mLambda = l;
    }

    public double getLambda() {
        return mLambda;
    }

    public void setGamma(double l) {
        mGamma = l;
    }

    public double getGamma() {
        return mGamma;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        FeatureInfo[] featureInfo = createFeatureInfo(data);
        Attribute target = data.classAttribute();
        mTargetStats = data.attributeStats(data.classIndex()).numericStats;

        Layer[] layers = new Layer[] {
            new FullyConnected(mNumTrees, mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumTrees, mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumHidden, target.isNominal() ? (target.numValues() - 1) : 1, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon)
        };

        StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
        options.gracePeriod = mBatchSize;
        options.lambda = mLambda;
        options.gamma = mGamma;

        mNetwork = new StreamingGradientTreeNetwork(featureInfo, options, mNumTrees, layers);
        mNetwork.randomlyInitialize(new Random(), 1.0 / Math.sqrt(mNumTrees));
        mObjective = target.isNominal() ? new SoftmaxCrossEntropy() : new SquaredError();

        for(int e = 0; e < mEpochs; e++) {
            for(int i = 0; i < data.numInstances(); i++) {
                int[] features = getFeatures(data.instance(i));

                double[] pred = mNetwork.predict(features);
                double[] groundTruth;
                
                if(target.isNominal()) {
                    groundTruth = new double[target.numValues()];
                    groundTruth = new double[target.numValues()];
                    groundTruth[(int)data.instance(i).classValue()] = 1.0;
                }
                else {
                    groundTruth = new double[1];
                    groundTruth[0] = (data.instance(i).classValue() - mTargetStats.mean) / mTargetStats.stdDev;
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
        pred[0] = pred[0] * mTargetStats.stdDev + mTargetStats.mean;

        return mObjective.transfer(pred);
    }
    
    private FeatureInfo[] createFeatureInfo(Instances insts) throws Exception {
        FeatureInfo[] featureInfo = new FeatureInfo[insts.numAttributes() - 1];
        int i = 0;

        for(int attInd = 0; attInd < insts.numAttributes(); attInd++) {
            if(attInd == insts.classIndex()) {
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
        
        int[] features = new int[inst.numAttributes() - 1];

        int i = 0;

        for(int attInd = 0; attInd < inst.numAttributes(); attInd++) {
            if(attInd == inst.classIndex()) {
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