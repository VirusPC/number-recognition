package neuralNetwork;

import java.io.Serializable;
import java.util.Random;

public class Neron implements Serializable {

    private static final long serialVersionUID = 33333333;

    private double inputValue;  // 单元输入值
    private double outputValue;  // 单元输出值
    private double delta;  // 偏差值，cost(i)对inputValue求导
    private double[] weights;  // 与该神经单元连接的左侧参数
    private double bias; // 偏置

    public Neron(){

    }


    public void setOutputValue(double outputValue){
        this.outputValue = outputValue;
    }



    public double[] getWeights(){
        return weights;
    }

    public void setWeights(double[] weights)
    {
        this.weights = weights;
    }

    /**
     * He初始化权重
     * @param neronsNumRight 与该神经元相连的左侧权重的数量
     */
    public void initialWeights(int neronsNumLeft, int neronsNumRight){
        double[] weights = new double[neronsNumLeft];
        Random random = new Random();

        for (int i=0; i <neronsNumLeft ; i++) {
            double r = random.nextGaussian();//*Math.sqrt(2/(double)n);
            weights[i] = r * Math.sqrt(2 / (double)(neronsNumLeft+1));
        }
        this.weights = weights;

        double bias = random.nextGaussian() * Math.sqrt(2/(double)(neronsNumLeft+1));
        this.bias = bias;
    }

    public double getOutputValue(){
        return outputValue;
    }

    public double getBias(){
        return bias;
    }

    public void setBias(double bias){
        this.bias = bias;
    }

    public double getDelta(){
        return this.delta;
    }

    public void setDelta(double delta){
        this.delta = delta;
    }

    public void resetDelta(){
        this.delta = 0;
    }



}
