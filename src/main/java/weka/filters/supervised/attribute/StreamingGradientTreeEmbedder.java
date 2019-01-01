package weka.filters.supervised.attribute;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import com.henrygouk.sgt.*;

import java.util.Random;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.experiment.Stats;
import weka.filters.SimpleBatchFilter;

public class StreamingGradientTreeEmbedder extends SimpleBatchFilter implements Serializable {

    private static final long serialVersionUID = 7719967089099105818L;

    protected int mDimensions = 32;

    protected double mLambda = 0.1;

    protected double mGamma = 1.0;

    protected int mGracePeriod = 200;

    protected int mBins = 64;

    protected int mEpochs = 20;

    protected double[] mMin;

    protected double[] mMax;

    protected StreamingGradientTreeCommittee mTrees;

    public int getDimensions() {
        return mDimensions;
    }

    public void setDimensions(int v) {
        mDimensions = v;
    }

    public int getEpochs() {
        return mEpochs;
    }

    public void setEpochs(int v) {
        mEpochs = v;
    }

    public int getGracePeriod() {
        return mGracePeriod;
    }

    public void setGracePeriod(int g) {
        mGracePeriod = g;
    }

    public double getLambda() {
        return mLambda;
    }

    public void setLambda(double l) {
        mLambda = l;
    }

    public double getGamma() {
        return mGamma;
    }

    public void setGamma(double g) {
        mGamma = g;
    }

    public int getBins() {
        return mBins;
    }

    public void setBins(int bins) {
        mBins = bins;
    }

    public String[] getOptions() {
        Vector<String> options = new Vector<String>();

        options.add("-D");
        options.add(Integer.toString(getDimensions()));

        options.add("-E");
        options.add(Integer.toString(getEpochs()));

        options.add("-G");
        options.add(Integer.toBinaryString(getGracePeriod()));

        options.add("-L");
        options.add(Double.toString(getLambda()));

        options.add("-Y");
        options.add(Double.toString(getGamma()));

        options.add("-B");
        options.add(Integer.toString(getBins()));

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[options.size()]);
    }

    public void setOptions(String[] options) throws Exception {

        setDimensions(Integer.parseInt(getOption(options, "D", 32)));
        setEpochs(Integer.parseInt(getOption(options, "E", 20)));
        setGracePeriod(Integer.parseInt(getOption(options, "G", 200)));
        setLambda(Double.parseDouble(getOption(options, "L", 0.1)));
        setGamma(Double.parseDouble(getOption(options, "Y", 1.0)));
        setBins(Integer.parseInt(getOption(options, "B", 64)));
    }

    protected String getOption(String[] allOptions, String optionName, Object defaultValue) throws Exception {
        String optionValue = Utils.getOption(optionName, allOptions);

		if(optionValue.length() == 0) {

			return defaultValue.toString();
        }
        else {
            return optionValue;
        }
    }

    @Override
    public String globalInfo() {
        return "Embeds instances using a committee of streaming gradient trees.";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		for(int i = 0; i < mDimensions; i++) {

			attributes.add(new Attribute("target" + Integer.toString(i)));
        }
        
        attributes.add((Attribute)inputFormat.classAttribute().copy());

		Instances newInstances = new Instances(inputFormat.relationName(), attributes, 0);
		newInstances.setClassIndex(attributes.size() - 1);

		return newInstances;
    }

    @Override
    protected Instances process(Instances instances) throws Exception {
        if(mTrees == null) {
            train(instances);
        }

        return embed(instances);
	}
    
    protected void train(Instances instances) throws Exception {
        Random rng = new Random(1);
        FeatureInfo[] featureInfo = createFeatureInfo(instances);
        StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
        options.gracePeriod = mGracePeriod;
        options.lambda = mLambda;
        options.gamma = mGamma;

        mTrees = new StreamingGradientTreeCommittee(featureInfo, options, mDimensions);
        mTrees.randomlyInitialize(rng, Math.sqrt(1.0 / mDimensions));

        int pairs = instances.numInstances() * mEpochs;

        for(int p = 0; p < pairs; p++) {
            int i = rng.nextInt(instances.numInstances());
            int j = i;

            while(j == i || (instances.get(i).classValue() == instances.get(j).classValue()) != (p % 2 == 0)) {
                j = rng.nextInt(instances.numInstances());
            }

            int[] xi = getFeatures(instances.instance(i));
            int[] xj = getFeatures(instances.instance(j));
            double[] zi = mTrees.predict(xi);
            double[] zj = mTrees.predict(xj);
            double w = euclidean(zi, zj);
            double y = instances.get(i).classValue() == instances.get(j).classValue() ? 1.0 : 0.0;

            GradHess[] grads = new GradHess[zi.length];

            for(int c = 0; c < grads.length; c++) {
                grads[c] = new GradHess();
                grads[c].gradient = y * (zi[c] - zj[c]) + (1.0 - y) * (w < 1.0 ? (zj[c] - zi[c]) : 0.0);
                grads[c].hessian = y + (1.0 - y) * (w < 1.0 ? -1.0 : 0.0);
            }

            mTrees.update(xi, grads);

            for(int c = 0; c < grads.length; c++) {
                grads[c].gradient = y * (zj[c] - zi[c]) + (1.0 - y) * (w < 1.0 ? (zi[c] - zj[c]) : 0.0);
                grads[c].hessian = y + (1.0 - y) * (w < 1.0 ? -1.0 : 0.0);
            }

            mTrees.update(xj, grads);
        }
    }

    protected Instances embed(Instances instances) throws Exception {
        Instances ret = determineOutputFormat(instances);

        for(int i = 0; i < instances.numInstances(); i++) {
            double[] newFeatures = mTrees.predict(getFeatures(instances.instance(i)));
            double[] vals = new double[ret.numAttributes()];

            for(int j = 0; j < newFeatures.length; j++) {
                vals[j] = newFeatures[j];
            }

            vals[vals.length - 1] = instances.get(i).classValue();

            ret.add(new DenseInstance(instances.instance(i).weight(), vals));
        }

        return ret;
    }

    private FeatureInfo[] createFeatureInfo(Instances insts) {
        FeatureInfo[] featureInfo = new FeatureInfo[insts.numAttributes() - 1];
        mMax = new double[featureInfo.length];
        mMin = new double[featureInfo.length];
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
                Stats stats = insts.attributeStats(attInd).numericStats;
                mMin[i] = stats.min;
                mMax[i] = stats.max;
            }

            i++;
        }

        return featureInfo;
    }

    private int[] getFeatures(Instance inst) {
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
            else {
                features[i] = (int)((double)mBins * ((inst.value(attInd) - mMin[i]) / (mMax[i] - mMin[i])));

                if(features[i] < 0) {
                    features[i] = 0;
                }
                else if(features[i] >= mBins) {
                    features[i] = mBins - 1;
                }
            }

            i++;
        }

        return features;
    }

    private double euclidean(double[] a, double[] b) {
        double ret = 0.0;

        for(int i = 0; i < a.length; i++) {
            ret += Math.pow(a[i] - b[i], 2.0);
        }

        return ret;
    }
}