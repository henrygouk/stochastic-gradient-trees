package weka.classifiers.meta;

import weka.classifiers.*;
import weka.core.*;
import weka.experiment.Stats;

public class StandardizedRegression extends SingleClassifierEnhancer implements BatchPredictor {

    private static final long serialVersionUID = 1L;

    private Stats mTargetStats;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        data = new Instances(data);
        mTargetStats = data.attributeStats(data.classIndex()).numericStats;
        int cind = data.classIndex();

        for(int i = 0; i < data.numInstances(); i++) {
            double t = (data.instance(i).value(cind) - mTargetStats.mean) / mTargetStats.stdDev;
            data.instance(i).setValue(cind, t);
        }

        getClassifier().buildClassifier(data);
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        double[] pred = getClassifier().distributionForInstance(inst);

        pred[0] = pred[0] * mTargetStats.stdDev + mTargetStats.mean;

        return pred;
    }

    @Override
    public double[][] distributionsForInstances(Instances data) throws Exception {
        double[][] preds = ((AbstractClassifier)getClassifier()).distributionsForInstances(data);

        for(int i = 0; i < preds.length; i++) {
            preds[i][0] = preds[i][0] * mTargetStats.stdDev + mTargetStats.mean;
        }

        return preds;
    }

    @Override
    public boolean implementsMoreEfficientBatchPrediction() {
        return true;
    }

}