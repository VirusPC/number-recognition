package neuralNetwork;

import java.io.Serializable;

public class ReluLayer extends Layer implements Serializable {

    /**
     * ReLU函数
     * @param x
     * @return
     */
    private double ReLU(double x){
        return Math.max(0, x);
    }

    /**
     * ReLU函数的导数
     * @param z
     * @return
     */
    private  double derivativeForReLU(double z){
        if(z>=0){
            return 1;
        }
        return 0;
    }

    @Override
    public void setInputValues(double[] inputValues) {
        for(int i=0; i<inputValues.length; i++){
            inputValues[i] = ReLU(inputValues[i]);
        }
        super.setOutputValues(inputValues);
    }

    @Override
    public double[] getDerivativesOfActivateFunction() {
        double[] derivatives = new double[super.getNeronNum()];
        double[] outputValues = super.getOutputValues();
        for(int i=0; i<super.getNeronNum(); i++){
            derivatives[i] = derivativeForReLU(outputValues[i]);
        }
        return derivatives;
    }

    @Override
    public double[] getDerivativesOfCostFunciton(double[] label) {
        return new double[0];
    }

    @Override
    public double getCost(double[] prediction, double[] label) {
        return 0;
    }

}
