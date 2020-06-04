package neuralNetwork;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Model implements Serializable {

    private static final long serialVersionUID = 11111111;

    private List<Layer> layers;
    private int[] shape;

    public Model(){
        layers = new ArrayList<Layer>();
    }

    public Model(List<Layer> layers){
        this.layers = layers;
        shape = new int[layers.size()];
        for(int i=0; i<shape.length; i++){
            shape[i] = layers.get(i).getNeronNum();
        }
    }

    public Model(int[] shape){
        this.shape = shape;
        buildStructure(shape);
        initialWeightsss();
        //initialWeightsssForTest();
    }

    /**
     * 创建模型结构
     * @param neronsNumEachLayer
     */
    public void buildStructure(int[] neronsNumEachLayer){
        layers = new ArrayList<>();

        // 除最后一层，其他层激活函数为ReLU
        for(int i=0; i<neronsNumEachLayer.length-1; i++){
            Layer reluLayer = new ReluLayer();
            for(int j=0; j<neronsNumEachLayer[i]; j++){
                reluLayer.addNeron(new Neron());
            }
            layers.add(reluLayer);
        }

        // 最后一层激活函数为softMax
        Layer softMaxLayer = new SoftMaxLayer();
        for(int j=0; j<neronsNumEachLayer[neronsNumEachLayer.length-1]; j++){
            softMaxLayer.addNeron(new Neron());
        }
        layers.add(softMaxLayer);

    }

    public void addLayer(Layer layer){
        layers.add(layer);
    }

    public Layer getLayer(int i){
        return layers.get(i);
    }

    public List<Layer> getLayers(){
        return layers;
    }

    public Layer getInputLayer(){
        return layers.get(0);
    }

    public Layer getOutputLayer(){
        return layers.get(layers.size()-1);
    }

    /**
     * 初始化权重
     */
    private void initialWeightsss(){

        // 输入层
        //layers.get(0).initialWeightss(0);

        // 遍历隐藏层和输出层
        for(int l=1; l<layers.size(); l++){
            // 每个神经元左侧的权重的个数
            int weightsNumLeft = layers.get(l-1).getNeronNum();
            // 为l层的每个神经元，设置位于l-1层到l层之间且与之相连接的权重。
            layers.get(l).initialWeightss(weightsNumLeft);
        }
    }


    /**
     * 获取权重
     * @return
     */
    public List<double[][]> getWeightsss(){
        List<double[][]> weightsss = new ArrayList<>();

        for(int i=1; i<layers.size(); i++){
            weightsss.add(layers.get(i).getWeightss());
        }
        return weightsss;
    }

    /**
     * 设置权重
     * @param weightsss
     */
    public void setWeights(List<double[][]> weightsss){
        for(int i=1; i<layers.size(); i++){
            layers.get(i).setWeightss(weightsss.get(i-1));
        }
    }

    /**
     * 获取偏置
     * @return
     */
    public List<double[]> getBiasss(){
        List<double[]> biasss = new ArrayList<>();
        for(int i=1; i<layers.size(); i++){
            biasss.add(layers.get(i).getBiass());
        }
        return biasss;
    }

    /**
     * 设置偏置
     * @param biasss
     */
    public void setBiasss(List<double[]> biasss){
        for(int i=1; i<layers.size(); i++){
            layers.get(i).setBiass(biasss.get(i-1));
        }
    }


    /**
     * 单样本单次正向传播
     */
    public void  forwardProp(double[] example){
        // 输入样本
        Layer inputLayer = layers.get(0);
        inputLayer.setOutputValues(example);
        // 通过所有的隐藏层和输出层计算输出结果
        for(int i=1; i<layers.size(); i++) {
            Layer layerLeft = layers.get(i-1);
            Layer layerRight = layers.get(i);
            double[] outputValuesLeft = layerLeft.getOutputValues();
            double[] inputValuesRight = new double[layerRight.getNeronNum()];
            double[][] weightss = layerRight.getWeightss();
            double[] biass = layerRight.getBiass();
            // 遍历该层所有的神经元，为它们求得输入值
            for(int j=0; j<layerRight.getNeronNum(); j++){
                double[] weights = weightss[j];
                for(int k=0; k<outputValuesLeft.length; k++){
                    inputValuesRight[j] += outputValuesLeft[k] * weights[k];
                }
                // 加上偏置项
                inputValuesRight[j] += biass[j];
            }
            layerRight.setInputValues(inputValuesRight);
        }
    }


    /**
     * 单样本单次反向传播，误差累加
     */
    public void backProp(double[] label){

        //设置输出层的误差
        Layer outputLayer = layers.get(layers.size()-1);
        double[] hypothesis = outputLayer.getOutputValues();
        double[] deltasInOutputLayer = new double[label.length];

        for(int i=0; i<label.length; i++){
            deltasInOutputLayer[i] = hypothesis[i]-label[i];
        }

        outputLayer.setDeltas(deltasInOutputLayer);

        // 反向遍历所有的隐藏层计算误差
        for(int i=layers.size()-2; i>0; i--){
            Layer layerLeft = layers.get(i);
            Layer layerRight = layers.get(i+1);

            double[] deltasRight = layerRight.getDeltas();
            double[] deltasLeft = new double[layerLeft.getNeronNum()];
            double[][] weightss = layerRight.getWeightss();
            double[] derivativesLeft = layerLeft.getDerivativesOfActivateFunction();

            // 计算左侧每个神经元的误差，累加
            for(int j=0; j<layerLeft.getNeronNum(); j++){
                double deltaLeft = 0;
                for(int k=0; k<weightss.length; k++){
                    double[] weights = weightss[k];
                    deltaLeft += deltasRight[k]*weights[j];
                }
                deltaLeft *= derivativesLeft[j];
                deltasLeft[j] = deltaLeft;
            }
            layerLeft.setDeltas(deltasLeft);
        }
    }


    /**
     * 获取dw
     * @return dw
     */
    public List<double[][]> getDwsss(){
       return getRegularizedDwsss(0);
    }

    public List<double[][]> getRegularizedDwsss(double lambda2){
        List<double[][]> dwsss = new ArrayList<>();
        List<double[][]> weightsss = getWeightsss();
        for(int i=1; i<layers.size(); i++){
            double[] outputValuesLeft = layers.get(i-1).getOutputValues();
            Layer layerRight = layers.get(i);
            double[][] dwss = new double[layerRight.getNeronNum()][];
            for(int j=0; j<layerRight.getNeronNum(); j++){
                double[] deltas = layerRight.getDeltas();
                double[] dws = new double[outputValuesLeft.length];
                for(int k=0; k<outputValuesLeft.length; k++){
                    double dr = lambda2*weightsss.get(dwsss.size())[j][k];
                    dws[k] = outputValuesLeft[k]*deltas[j] + dr;
                }
                dwss[j] = dws;
            }
            dwsss.add(dwss);
        }
        return dwsss;
    }

    /**
     * bias无需正则化
     * @return dbsss
     */
    public List<double[]> getDbss(){
        List<double[]> dbss = new ArrayList<>();
        for(int i=1; i<layers.size(); i++){
            Layer layerRight = layers.get(i);
            double[] deltas = layerRight.getDeltas();
            double[] dbs = new double[deltas.length];

            for(int j=0; j<deltas.length; j++){
                dbs[j] = deltas[j];
            }

            dbss.add(dbs);
        }
        return dbss;
    }

    /**
     * 获取cost
     * @param examples
     * @param labels
     * @return
     */
    public double getCost(double[][] examples, double[][] labels){
        double cost = 0;

        for(int i=0; i<examples.length; i++){
            double[] example = examples[i];
            double[] label = labels[i];
            double[] prediction = this.predict(example);
            cost += this.getOutputLayer().getCost(prediction, label);
        }

        cost = cost/examples.length;
        return cost;
    }


    public int getPredictedClassOrder(double[] example){
        double[] prediction = predict(example);
        double maxValue = prediction[0];
        int maxOrder = 0;
        for(int i=1; i<prediction.length; i++){
            if(prediction[i]>maxValue){
                maxValue = prediction[i];
                maxOrder = i;
            }
        }
        return maxOrder;
    }


    /**
     * 精度
     * @param examples
     * @param labels
     * @return
     */
    public double getAccuracy(double[][] examples, double[][] labels) {
        double accuracy = 0;
        double total = examples.length;
        double right = 0;
        for(int i=0; i<total; i++){
            double[] example = examples[i];
            double[] label =  labels[i];
            int orderNum = getPredictedClassOrder(example);
            if(label[orderNum] == 1){
                right++;
            }
        }
        accuracy = right/total;
        return accuracy;
    }

    /**
     * 误差
     * @param examples
     * @param labels
     * @return
     */
    public double getError(double[][] examples, double[][] labels){
        return 1-getAccuracy(examples, labels);
    }

    /**
     * 查准率
     * 预测
     * @param examples
     * @param labels
     * @return
     */
    public double getPrecision(double[][] examples, double[][]labels){
        double precision = 0;
        return precision;
    }

    /**
     *
     * @param examples
     * @param labels
     * @return
     */
    public double getRecall(double[][] examples, double[][] labels){
        double recall = 0;
        return recall;
    }

    /**
     * F_beta 度量
     * beta>1时更关注查全率，beta<1时更关注查准率
     * @param examples
     * @param labels
     * @param beta
     * @return
     */
    public double getFBetaScore(double[][] examples, double[][] labels, double beta){
        double precision = this.getPrecision(examples, labels);
        double recall = this.getRecall(examples, labels);
        double fBetaScore = ((1+Math.pow(beta, 2))*precision*recall) / (Math.pow(beta, 2)*precision+recall);
        return fBetaScore;
    }

    /**
     * F_1 度量
     * beta为1的F_beta度量
     * @param examples
     * @param labels
     * @return
     */
    public double getF1Score(double[][] examples, double[][] labels){
        return this.getFBetaScore(examples, labels, 1);
    }

    public double getMacroPrecision(double[][] examples, double[][] labels){
        double macroP = 0;
        return macroP;
    }

    public double getMacroRecall(double[][] examples, double[][] labels){
        double macroRecall = 0;
        return macroRecall;
    }

    public double getMacroF1(double[][] examples, double[][] labels){
        double macroF1 = 0;
        return macroF1;
    }

    public int[] getShape(){
        return shape;
    }





    /**
     * 预测
     * @param x 输入的数据
     * @return 类别
     */
    public double[] predict(double[] x){

        // 输入数据
        Layer inputLayer = layers.get(0);
        inputLayer.setInputValues(x);

        // 正向传播
        forwardProp(x);

        // 找输出层输出值最大的神经单元的序号。
        Layer outputLayer = layers.get(layers.size()-1);

        return outputLayer.getOutputValues();
    }


}
