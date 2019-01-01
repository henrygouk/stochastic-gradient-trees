package moa.classifiers.trees.sgt;

import java.io.Serializable;

import com.henrygouk.sgt.*;
import com.yahoo.labs.samoa.instances.*;

public class AttributeDiscretizer implements Serializable {
    protected int mBins;
    protected FeatureInfo[] mFeatureInfo;
    protected double[] mMin;
    protected double[] mMax;

    public AttributeDiscretizer(int bins) {
        mBins = bins;
    }

    public void observe(Instance inst) {
        if(mFeatureInfo == null) {
            createFeatureInfo(inst);
        }

        for(int i = 0; i < inst.numInputAttributes(); i++) {
            if(!Double.isNaN(inst.valueInputAttribute(i))) {
                mMin[i] = Math.min(mMin[i], inst.valueInputAttribute(i));
                mMax[i] = Math.max(mMax[i], inst.valueInputAttribute(i));
            }
        }
    }

    public int[] getFeatures(Instance inst) {
        int[] features = new int[inst.numInputAttributes()];

        for(int i = 0; i < features.length; i++) {
            if(Double.isNaN(inst.valueInputAttribute(i))) {
                features[i] = -1;
            }
            else if(inst.inputAttribute(i).isNominal()) {
                features[i] = (int)inst.valueInputAttribute(i);
            }
            else {
                features[i] = (int)((double)mBins * ((inst.valueInputAttribute(i) - mMin[i]) / (mMax[i] - mMin[i])));

                if(features[i] < 0) {
                    features[i] = 0;
                }
                else if(features[i] >= mBins) {
                    features[i] = mBins - 1;
                }
            }
        }

        return features;
    }

    public FeatureInfo[] getFeatureInfo() {
        return mFeatureInfo;
    }

    protected void createFeatureInfo(Instance inst) {
        mFeatureInfo = new FeatureInfo[inst.numInputAttributes()];

        for(int i = 0; i < mFeatureInfo.length; i++) {
            mFeatureInfo[i] = new FeatureInfo();
            mMax = new double[mFeatureInfo.length];
            mMin = new double[mFeatureInfo.length];

            if(inst.inputAttribute(i).isNominal()) {
                mFeatureInfo[i].type = FeatureType.nominal;
                mFeatureInfo[i].categories = inst.inputAttribute(i).numValues();
            }
            else if(inst.inputAttribute(i).isNumeric()) {
                mFeatureInfo[i].type = FeatureType.ordinal;
                mFeatureInfo[i].categories = mBins;
            }
        }
    }
}