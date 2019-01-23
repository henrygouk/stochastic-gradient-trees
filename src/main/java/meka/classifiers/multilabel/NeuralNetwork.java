package meka.classifiers.multilabel;

import java.util.Arrays;

import com.henrygouk.sgt.*;
import com.henrygouk.sgt.neural.*;

import weka.core.Instance;
import weka.core.Instances;

public class NeuralNetwork extends AbstractMultiLabelClassifier {

    private static final long serialVersionUID = 1L;
    protected Layer[] mLayers;
    protected int mNumHidden = 100;
    protected int mBatchSize = 100;
    protected double mLearningRate = 0.001;
    protected double mBeta1 = 0.9;
    protected double mBeta2 = 0.999;
    protected double mEpsilon = 1E-8;
    protected int mEpochs = 20;
    protected Objective mObjective;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        mLayers = new Layer[] {
            new FullyConnected(data.numAttributes() - data.classIndex(), mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumHidden, mNumHidden, mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon),
            new RectifiedLinearUnit(),
            new FullyConnected(mNumHidden, data.classIndex(), mBatchSize, mLearningRate, mBeta1, mBeta2, mEpsilon)
        };

        mObjective = new BinaryCrossEntropy();

        for(int e = 0; e < mEpochs; e++) {
            for(int i = 0; i < data.numInstances(); i++) {
                double[] groundTruth = new double[data.classIndex()];
                double[] pred = distributionForInstance(data.instance(i));
                groundTruth = new double[data.classIndex()];

                for(int l = 0; l < data.classIndex(); l++) {
                    groundTruth[l] = data.instance(i).value(l);
                }

                GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, pred);

                double[][] activations = new double[mLayers.length + 1][];
                activations[0] = Arrays.copyOfRange(data.instance(i).toDoubleArray(), data.classIndex(), data.numAttributes());
                
                for(int l = 0; l < mLayers.length; l++) {
                    activations[l + 1] = mLayers[l].predict(activations[l]);
                }

                for(int l = mLayers.length - 1; l >= 0; l--) {
                    gradHess = mLayers[l].update(activations[l], gradHess);
                }
            }
        }
    }

    @Override
	public double[] distributionForInstance(Instance inst) throws Exception {
		double[][] activations = new double[mLayers.length + 1][];
        activations[0] = Arrays.copyOfRange(inst.toDoubleArray(), inst.classIndex(), inst.numAttributes());
        
        for(int i = 0; i < mLayers.length; i++) {
            activations[i + 1] = mLayers[i].predict(activations[i]);
        }

        return activations[activations.length - 1];
	}

}