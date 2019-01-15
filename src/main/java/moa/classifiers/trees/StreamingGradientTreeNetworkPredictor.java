package moa.classifiers.trees;

import java.util.Random;

import com.github.javacliparser.*;
import com.henrygouk.sgt.*;
import com.henrygouk.sgt.neural.*;

public class StreamingGradientTreeNetworkPredictor extends StreamingGradientTreePredictor {

    private static final long serialVersionUID = 1L;

    public IntOption numTrees = new IntOption("numTrees", 'T',
        "The number of trees in the SGT layer.", 10, 0, Integer.MAX_VALUE);
    
    public StringOption hiddenUnits = new StringOption("hiddenUnits", 'H',
        "A comma separated list of hidden layer sizes.", "100,100");
    
    public FloatOption learningRate = new FloatOption("learningRate", 'A',
        "The learning rate for Adam.", 0.9, 0.0, 1.0);

    public FloatOption beta1 = new FloatOption("beta1", '1',
        "The momentum rate for Adam.", 0.9, 0.0, 1.0);
    
    public FloatOption beta2 = new FloatOption("beta2", '2',
        "The second order momentum rate for Adam.", 0.999, 0.0, 1.0);
    
    public FloatOption epsilon = new FloatOption("epsilon", 'E',
        "The epsilon value for adam", 1E-8, 0.0, Double.MAX_VALUE);

    @Override
    protected MultiOutputLearner createTrees(FeatureInfo[] featureInfo, StreamingGradientTreeOptions options, int numOutputs) {
        int prevSize = numTrees.getValue();
        int[] hiddenSizes = {100, 100};
        Layer[] layers = new Layer[hiddenSizes.length * 2 + 1];

        for(int i = 0; i < hiddenSizes.length; i++) {
            layers[i * 2] = new FullyConnected(prevSize, hiddenSizes[i], options.gracePeriod, learningRate.getValue(),
                beta1.getValue(), beta2.getValue(), epsilon.getValue());
            layers[i * 2 + 1] = new RectifiedLinearUnit();
            prevSize = hiddenSizes[i];
        }

        layers[layers.length - 1] = new FullyConnected(prevSize, numOutputs, options.gracePeriod, learningRate.getValue(),
            beta1.getValue(), beta2.getValue(), epsilon.getValue());
        
        StreamingGradientTreeNetwork ret = new StreamingGradientTreeNetwork(featureInfo, options, numTrees.getValue(), layers);
        ret.randomlyInitialize(new Random(), 1.0 / Math.sqrt(numTrees.getValue()));

        return ret;
    }
}