package neuralNetwork;

import java.io.Serializable;

public class SoftMaxLayer extends Layer implements Serializable {

    private double[] softMax(double[] vector){
        //double[] outputValues = new double[inputValues.length];
        double sum = 0;
        for(int i=0; i<vector.length; i++){
            vector[i] = Math.pow(Math.E, vector[i]);
            sum += vector[i];
        }
        for(int i=0; i<vector.length; i++){
            vector[i] = vector[i]/sum;
        }
        return vector;
    }

    @Override
    public void setInputValues(double[] inputValues){
        super.setOutputValues(softMax(inputValues));
    }

    @Override
    public double[] getDerivativesOfActivateFunction() {
        return null;
    }



    /**
     * categorical cross entropy
     * @param label
     * @return
     */
    @Override
    public double getCost(double[] prediction, double[] label) {
        double cost = 0;
        for(int j=0; j<label.length; j++){
            if(label[j]==1) {
                cost = -Math.log(prediction[j]);
                break;
            }
        }
        return cost;
    }

    /**
     * 分类交叉熵对输出层的inputValue求导
     * @return
     */
    @Override
    public double[] getDerivativesOfCostFunciton(double[] label) {
        double[] prediction = this.getOutputValues();
        for(int i=0 ;i<label.length; i++){
            prediction[i] -= label[i];
        }
        return prediction;
    }


}
