package moa.classifiers.trees;

import java.io.Serializable;
import java.util.stream.IntStream;

import com.github.javacliparser.*;
import com.henrygouk.sgt.*;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.trees.sgt.*;
import moa.core.Measurement;

public class StreamingGradientTreePredictor extends AbstractClassifier implements Serializable, MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

    protected StreamingGradientTreeCommittee mTrees;

    protected AttributeDiscretizer mDiscretizer;

    protected int mInstances;

    protected Objective mObjective;

    public FloatOption delta = new FloatOption("delta", 'D',
        "The confidence level used when performing the hypothesis tests.", 1E-7, 0.0, 1.0);
    
    public FloatOption lambda = new FloatOption("lambda", 'L',
        "Regularisation parameter that can be used to influence the magnitude of updates.", 0.1, 0.0, Double.POSITIVE_INFINITY);
    
    public FloatOption gamma = new FloatOption("gamma", 'Y',
        "The loss incurred from adding a new node to the tree.", 1.0, 0.0, Double.POSITIVE_INFINITY);

    public IntOption gracePeriod = new IntOption("gracePeriod", 'G',
        "The number of instances to observe between searches for new splits.", 200, 0, Integer.MAX_VALUE);
    
    public IntOption warmStart = new IntOption("warmStart", 'W',
        "The number of instances used to estimate bin boundaries for numeric values.", 1000, 0, Integer.MAX_VALUE);

    public IntOption bins = new IntOption("bins", 'B',
        "The number of bins to be used for discretizing numeric attributes.", 64, 0, Integer.MAX_VALUE);

    @Override
    public String getPurposeString() {
        return "Trains a single Streaming Gradient Tree for regression, or a committe for classification.";
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
    
    @Override
    public void resetLearningImpl() {
        mTrees = null;
        mDiscretizer = new AttributeDiscretizer(bins.getValue());
        mInstances = 0;
    }

    @Override
    public int measureByteSize() {
        return 0;
    }

    public void trainOnInstanceImpl(Instance inst) {
        mInstances++;

        if(mInstances <= warmStart.getValue()) {
            mDiscretizer.observe(inst);
            return;
        }

        Attribute target = inst.classAttribute();

        if(mTrees == null) {
            FeatureInfo[] featureInfo = mDiscretizer.getFeatureInfo();
            StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
            options.delta = delta.getValue();
            options.gracePeriod = gracePeriod.getValue();
            options.lambda = lambda.getValue();
            options.gamma = gamma.getValue();

            if(target.isNominal()) {
                mTrees = new StreamingGradientTreeCommittee(featureInfo, options, target.numValues() - 1);
                mObjective = new SoftmaxCrossEntropy();
            }
            else {
                mTrees = new StreamingGradientTreeCommittee(featureInfo, options, 1);
                mObjective = new SquaredError();
            }
        }

        int[] features = mDiscretizer.getFeatures(inst);
        double[] groundTruth;
        double[] raw = mTrees.predict(features);

        if(target.isNominal()) {
            groundTruth = new double[target.numValues()];
            groundTruth[(int)inst.classValue()] = 1.0;
        }
        else {
            groundTruth = new double[] {inst.classValue()};
        }

        GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, raw);
        mTrees.update(features, gradHess);
    }

    public double[] getVotesForInstance(Instance inst) {
        if(mTrees == null) {
            if(inst.classAttribute().isNominal()) {
                return new double[inst.classAttribute().numValues()];
            }
            else {
                return new double[1];
            }
        }

        int[] features = mDiscretizer.getFeatures(inst);
        double[] raw = mTrees.predict(features);
        
        return mObjective.transfer(raw);
    }

    @Override
    public void getModelDescription(StringBuilder in, int indent) {
        //
    }

    @Override
    public Measurement[] getModelMeasurementsImpl() {
        double nodes = mTrees.getNumNodes();
        double splits = mTrees.getNumSplits();
        double updates = mTrees.getNumNodeUpdates();
        double maxDepth = mTrees.getMaxDepth();

        return new Measurement[] {
            new Measurement("nodes", nodes),
            new Measurement("splits", splits),
            new Measurement("node updates", updates),
            new Measurement("max depth", maxDepth)
        };
    }
}