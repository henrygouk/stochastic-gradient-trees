package weka.classifiers.trees;

import java.util.Random;

import com.henrygouk.sgt.*;

import scala.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class StochasticGradientTree extends AbstractClassifier {

    private static final long serialVersionUID = 3601534653937503693L;
    protected int mBins = 64;
    protected StreamingGradientTree mTree;
    protected int mBatchSize = 1;
    protected int mEpochs = 20;
    protected Objective mObjective;
    protected Discretize mDiscretize;
    protected double mLambda = 0.1;
    protected double mGamma = 1.0;
    protected FeatureInfo[] mFeatureInfo;

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
        if(data.numAttributes() == 3 && data.attribute(1).isRelationValued()) {
            buildMultiInstanceClassifier(data);
        }
        else {
            buildPropositionalClassifier(data);
        }
    }

    private void buildPropositionalClassifier(Instances data) throws Exception {
        mFeatureInfo = createFeatureInfo(data);
        mObjective = new SoftmaxCrossEntropy();
        StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
        options.gracePeriod = mBatchSize;
        options.lambda = mLambda;
        options.gamma = mGamma;

        mTree = new StreamingGradientTree(mFeatureInfo, options);

        for(int e = 0; e < mEpochs; e++) {
            for(int i = 0; i < data.numInstances(); i++) {
                double[] pred = new double[]{mTree.predict(getFeatures(data.instance(i)))};
                double[] target = new double[]{data.instance(i).classValue()};
                GradHess[] gradHess = mObjective.computeDerivatives(target, pred);
                mTree.update(getFeatures(data.instance(i)), gradHess[0]);
            }
        }
    }

    private void buildMultiInstanceClassifier(Instances data) throws Exception {
        Instances allInstances = new Instances(data.attribute(1).relation());

        for(int b = 0; b < data.numInstances(); b++) {
            Instances bag = data.instance(b).relationalValue(1);

            for(int i = 0; i < bag.numInstances(); i++) {
                allInstances.add(bag.instance(i));
            }
        }

        mFeatureInfo = createFeatureInfo(allInstances);
        mObjective = new SoftmaxCrossEntropy();

        StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
        options.gracePeriod = mBatchSize;
        options.lambda = mLambda;
        options.gamma = mGamma;

        mTree = new StreamingGradientTree(mFeatureInfo, options);

        for(int e = 0; e < mEpochs; e++) {
            for(int b = 0; b < data.numInstances(); b++) {
                Instances bag = data.instance(b).relationalValue(1);
                double bagLabel = data.instance(b).classValue();
                double topScore = Double.NEGATIVE_INFINITY;
                int topScoreInd = -1;

                for(int i = 0; i < bag.numInstances(); i++) {
                    double score = mTree.predict(getFeatures(bag.instance(i)));
                    
                    if(score > topScore) {
                        topScore = score;
                        topScoreInd = i;
                    }
                }

                double[] pred = new double[]{topScore};
                double[] target = new double[]{bagLabel};

                GradHess[] gradHess = mObjective.computeDerivatives(target, pred);
                GradHess nullGradHess = new GradHess(0.0, 0.0);

                for(int i = 0; i < bag.numInstances(); i++) {
                    if(i == topScoreInd) {
                        mTree.update(getFeatures(bag.instance(i)), gradHess[0]);
                    }
                    else {
                        mTree.update(getFeatures(bag.instance(i)), nullGradHess);
                    }
                }
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance inst) {
        if(inst.attribute(1).isRelationValued()) {
            Instances bag = inst.relationalValue(1);
            double topScore = Double.NEGATIVE_INFINITY;

            for(int i = 0; i < bag.numInstances(); i++) {
                int[] features = getFeatures(bag.instance(i));
                topScore = Math.max(topScore, mTree.predict(features));
            }

            double[] pred = new double[]{topScore};
            pred = mObjective.transfer(pred);
            return new double[]{pred[1], pred[0]};
        }
        else {
            double[] logit = new double[]{mTree.predict(getFeatures(inst))};
            double[] pred = mObjective.transfer(logit);
            return new double[]{pred[1], pred[0]};
        }
    }
    
    private FeatureInfo[] createFeatureInfo(Instances insts) throws Exception {
        FeatureInfo[] featureInfo = new FeatureInfo[insts.numAttributes() - (insts.classIndex() >= 0 ? 1 : 0)];
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
        mDiscretize.setUseEqualFrequency(false);
        mDiscretize.setBins(mBins);
        mDiscretize.setInputFormat(insts);
        Filter.useFilter(insts, mDiscretize);

        return featureInfo;
    }

    private int[] getFeatures(Instance inst) {
        mDiscretize.input(inst);
        mDiscretize.batchFinished();
        inst = mDiscretize.output();
        
        int[] features = new int[inst.numAttributes() - (inst.classIndex() >= 0 ? 1 : 0)];

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
