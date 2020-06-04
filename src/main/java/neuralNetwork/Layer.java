package neuralNetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public abstract class Layer implements Serializable {


    private static final long serialVersionUID = 22222222;

    private List<Neron> nerons;

    public abstract double getCost(double[] prediction, double[] label);
    public abstract double[] getDerivativesOfCostFunciton(double[] label);
    public abstract double[] getDerivativesOfActivateFunction();

    public Layer(){
        nerons = new ArrayList<Neron>();
    }

    public Layer(List<Neron> nerons){
        this.nerons = nerons;
    }

    public void addNeron(Neron neron){
        nerons.add(neron);
    }

    public int getNeronNum(){
        return nerons.size();
    }


    /**
     * 为该层的所有单元设置输入值
     * @param inputValues
     */
    public abstract void setInputValues(double[] inputValues);

    /**
     * 为该层的所有单元设置输出值
     * @param outputValues
     */
    public void setOutputValues(double[] outputValues){
        for(int i=0; i<nerons.size(); i++){
            Neron neron = nerons.get(i);
            neron.setOutputValue(outputValues[i]);
        }
    }


    /**
     * 获取该层神经单元的输出
     * @return
     */
    public double[] getOutputValues(){
        double[] outputValues = new double[nerons.size()];

        for(int i=0; i<nerons.size(); i++){
            outputValues[i] = nerons.get(i).getOutputValue();
        }

        return outputValues;
    }


    public double[] getDeltas(){
        double[] deltas = new double[nerons.size()];
        for(int i=0; i<deltas.length; i++){
            deltas[i] = nerons.get(i).getDelta();
        }
        return deltas;
    }

    public void setDeltas(double[] deltas){
        for(int i=0; i<deltas.length; i++){
            nerons.get(i).setDelta(deltas[i]);
        }
    }

    /**
     * 将误差置0
     */
    public void resetDelta(){
        for(Neron neron: nerons){
            neron.resetDelta();
        }
    }

    /**
     * 初始化该层左侧权重
     */
    public void initialWeightss(int neronsNumLeft){
        for(Neron neron: nerons) {
            neron.initialWeights(neronsNumLeft, nerons.size());
        }
    }

    /**
     * 设置该层权重
     * @param weightss
     */
    public void setWeightss(double[][] weightss){
        for(int i=0; i<nerons.size(); i++){
            nerons.get(i).setWeights(weightss[i]);
        }
    }

    /**
     * 获取该层权重
     * @return
     */
    public double[][] getWeightss(){
        double[][] weightss = new double[nerons.size()][];
        for(int i=0; i<nerons.size(); i++){
            weightss[i] = nerons.get(i).getWeights();
        }
        return weightss;
    }

    /**
     * 设置该层偏差
     * @param biass
     */
    public void setBiass(double[] biass){
        for(int i=0; i<nerons.size(); i++){
            nerons.get(i).setBias(biass[i]);
        }
    }

    /**
     * 获取该层偏差
     * @return
     */
    public double[] getBiass(){
        double[] biass = new double[nerons.size()];
        for(int i=0; i<nerons.size(); i++){
            biass[i] = nerons.get(i).getBias();
        }
        return biass;
    }



}
