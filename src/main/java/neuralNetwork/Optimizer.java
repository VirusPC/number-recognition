package neuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Optimizer {
    private Model model;

    public void gradientDescent(){}

    public Optimizer(Model model){
        this.model = model;
    }


    /**
     * 对部分参数采用默认值的自适应矩估计算法
     * @param examples 样本
     * @param labels 标签
     * @param itr 迭代次数
     * @param batchSize mini-batch的大小
     * @param learningRate 学习速率
     */
    public Model Adam(double[][] examples, double[][] labels, int itr, int batchSize, double learningRate, double lambda2){
        // 对部分超参数采用默认值
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 0.00000001;
        double decayRate = 0.01;
        model = Adam(examples, labels, itr, batchSize, learningRate, decayRate, beta1, beta2, epsilon, lambda2);

        return model;
    }

    /**
     * 自适应矩估计优化算法
     * @param examples 样本
     * @param labels 标签
     * @param itr 迭代次数
     * @param batchSize mini-batch的大小
     * @param learningRate0 初始学习速率
     * @param beta1 动量梯度下降超参数
     * @param beta2 均方根反向传播超参数
     * @param epsilon 保证分母不为0
     */
    public Model Adam(double[][] examples, double[][] labels, int itr, int batchSize, double learningRate0,
                      double decayRate, double beta1, double beta2, double epsilon, double lambda2){
        int[] shape = model.getShape();
        System.out.println("cost: "+model.getCost(examples, labels));
        // 进行itr次迭代
        for(int i=0; i<itr; i++){
            double startTime = System.currentTimeMillis();
            double learningRate = getDecayedLearningRate(learningRate0, decayRate, i);

            // 在每个mini-batch上进行迭代,每循环一次更新一次权重和偏差
            // epoch 从1开始计数。
            for(int epoch=1; epoch<examples.length/batchSize+1; epoch++){
                // 初始化
                List<double[][]> v_dwsss = this.getDwsssWithZero(shape);
                List<double[][]> s_dwsss = this.getDwsssWithZero(shape);
                List<double[]> v_dbss = this.getDbssWithZero(shape);
                List<double[]> s_dbss = this.getDbssWithZero(shape);
                List<double[][]> accedDwsss = this.getDwsssWithZero(shape);
                List<double[]> accedDbss = this.getDbssWithZero(shape);

                int from = batchSize*(epoch-1);
                int to  = Math.min(batchSize*epoch, examples.length);
                // mini-batch中的样本总数
                int m = to-from;
                for(int k=from; k<to; k++){
                    model.forwardProp(examples[k]);
                    model.backProp(labels[k]);
                    List<double[][]> dwsss = model.getRegularizedDwsss(lambda2);
                    List<double[]> dbss = model.getDbss();

                    accedDwsss = this.getSumOfDwsss(accedDwsss, dwsss);
                    accedDbss = this.getSumOfDbss(accedDbss, dbss);


                }
                accedDwsss = this.regularizeDwsss(accedDwsss, m);
                accedDbss = this.regularizeDbss(accedDbss, m);

                v_dwsss = this.getCorrectedFirstMomentEstimateOfDwsss(v_dwsss, accedDwsss, beta1, epoch);
                v_dbss = this.getCorrectedFirstMomentEstimateOfDbss(v_dbss, accedDbss, beta1, epoch);
                s_dwsss = this.getCorrectedSecondRawMomentEstimateOfDwsss(s_dwsss, accedDwsss, beta2, epoch);
                s_dbss = this.getCorrectedSecondRawMomentEstimateOfDbss(s_dbss, accedDbss, beta2, epoch);
                model.setWeights(this.updateWeightsss(model.getWeightsss(), v_dwsss, s_dwsss, learningRate, epsilon));
                model.setBiasss(this.updateBiasss(model.getBiasss(), v_dbss, s_dbss, learningRate, epsilon));
            }
            double endTime = System.currentTimeMillis();
            double elapsedTime = (endTime-startTime)/1000;
            System.out.println("Iteration: "+ (i+1) +"    Cost: "+model.getCost(examples, labels)+
                    "    elapsedTime: "+elapsedTime+"s");
        }
        return model;
    }

    public List<double[][]> getDwsssWithZero(int[] shape){
        ArrayList<double[][]> zerosss = new ArrayList<>();
        //int[] shape = model.getShape();

        for(int i=1; i<shape.length; i++){
            double[][] zeross = new double[shape[i]][shape[i-1]];
            zerosss.add(zeross);
        }
        return zerosss;
    }

    public List<double[]> getDbssWithZero(int[] shape){
        ArrayList<double[]> zeross = new ArrayList<>();

        //int[] shape = model.getShape();

        for(int i=1; i<shape.length; i++){
            double[] zeros = new double[shape[i]];
            zeross.add(zeros);
        }

        return zeross;
    }

    public List<double[][]> getSumOfDwsss(List<double[][]> dwsss1, List<double[][]> dwsss2){
        for(int i=0; i<dwsss1.size(); i++){
            double[][] dwss1 = dwsss1.get(i);
            double[][] dwss2 = dwsss2.get(i);
            for(int j=0; j<dwss1.length; j++){
                double[] dws1 = dwss1[j];
                double[] dws2 = dwss2[j];
                for(int k=0; k<dws1.length; k++){
                    dws1[k] += dws2[k];
                }
            }
        }
        return dwsss1;
    }

    public List<double[]> getSumOfDbss(List<double[]> dbss1, List<double[]> dbss2){
        for(int i=0; i<dbss1.size(); i++){
            double[] dbs1 = dbss1.get(i);
            double[] dbs2 = dbss2.get(i);
            for(int j=0; j<dbs1.length; j++){
                dbs1[j] = dbs1[j]+dbs2[j];
            }
        }
        return dbss1;
    }

    public List<double[][]> regularizeDwsss(List<double[][]> dwsss, double m){
        for(int i=0; i<dwsss.size(); i++){
            double[][] dwss = dwsss.get(i);
            for(int j=0; j<dwss.length; j++){
                double[] dws = dwss[j];
                for(int k=0; k<dws.length; k++){
                    dws[k] = dws[k]/m;
                }
            }
        }
        return dwsss;
    }

    public List<double[]> regularizeDbss(List<double[]> dbss, double m){
        for(int i=0; i<dbss.size(); i++){
            double[] dbs = dbss.get(i);
            for(int j=0; j<dbs.length; j++){
                dbs[j] = dbs[j]/m;
            }
        }
        return dbss;
    }

    /**
     * 求有偏一阶矩估计
     * v_dw := beta * v_dw + (1 - beta) * dw
     * @param dwsss
     * @return
     */
    public List<double[][]> getFirstMomentEstimateOfDwsss(List<double[][]> v_dwsss, List<double[][]> dwsss, double beta){
        return v_dwsss;
    }

    /**
     * 求有偏二阶原始矩估计
     * s_dw := beta * s_dw + (1 - beta) * dw^2
     * @param dwsss
     * @return
     */
    public List<double[][]> getSecondRawMomentEstimateOfDwsss(List<double[][]> s_dwsss, List<double[][]> dwsss, double beta){
        return s_dwsss;
    }

    /**
     * 求有偏一阶矩估计
     * v_db = beta * v_db + (1-beta) *  db
     * @param v_dbss
     * @param dbss
     * @return
     */
    public List<double[]> getFirstMomentEstimateOfDbss(List<double[]> v_dbss, List<double[]> dbss, double beta){
        return v_dbss;
    }

    /**
     * 求有偏二阶原始矩估计
     * s_db = beta * s_db + (1-beta) *  db^2
     * @param s_dbss
     * @param dbss
     * @return
     */
    public List<double[]> getSecondRawMomentEstimateofDbss(List<double[]> s_dbss, List<double[]> dbss, double beta){
        return s_dbss;
    }

    /**
     * 获取经偏差修正后的一阶矩估计
     * v_dw = [beta*v_dw + (1-beta)*dw] / (1-beta^epoch)
     * @param dwsss
     * @param epoch
     * @return
     */
    public List<double[][]> getCorrectedFirstMomentEstimateOfDwsss(List<double[][]> v_dwsss, List<double[][]> dwsss, double beta, int epoch){
        for(int i=0; i<v_dwsss.size(); i++){
            double[][] v_dwss = v_dwsss.get(i);
            double[][] dwss = dwsss.get(i);

            for(int j=0; j<v_dwss.length; j++){
                double[] v_dws = v_dwss[j];
                double[] dws = dwss[j];

                for(int k=0; k<v_dws.length; k++){
                    double v_dw = v_dws[k];
                    double dw = dws[k];
                    v_dws[k] = this.getCorrectedFirstMoment(v_dw, dw, beta, epoch);
                }

            }

        }
        return v_dwsss;
    }

    /**
     * 获取经偏差修正后的一阶矩估计
     * v_db = [beta*v_db + (1-beta)*db] / (1-beta^epoch)
     * @param dbss
     * @param epoch
     * @return
     */
    public List<double[]> getCorrectedFirstMomentEstimateOfDbss(List<double[]> v_dbss, List<double[]> dbss, double beta, int epoch){

        for(int i=0; i<v_dbss.size(); i++){
            double[] v_dbs = v_dbss.get(i);
            double[] dbs = dbss.get(i);

            for(int j=0; j<v_dbs.length; j++){
                double v_db = v_dbs[j];
                double db = dbs[j];
                v_dbs[j] = this.getCorrectedFirstMoment(v_db, db, beta, epoch);
            }

        }
        return v_dbss;
    }

    /**
     * 二阶矩估计
     * s_dw = [beta*s_dw + (1-beta)*dw^2]/(1-beta^epoch)
     * @param s_dwsss
     * @param dwsss
     * @param beta
     * @param epoch
     * @return
     */
    public List<double[][]> getCorrectedSecondRawMomentEstimateOfDwsss(List<double[][]> s_dwsss, List<double[][]> dwsss, double beta, int epoch){
        for(int i=0; i<s_dwsss.size(); i++){
            double[][] s_dwss = s_dwsss.get(i);
            double[][] dwss = dwsss.get(i);

            for(int j=0; j<s_dwss.length; j++){
                double[] s_dws = s_dwss[j];
                double[] dws = dwss[j];

                for(int k=0; k<s_dws.length; k++){
                    double v_dw = s_dws[k];
                    double dw = dws[k];
                    s_dws[k] = this.getCorrectedSecondRawMoment(v_dw, dw, beta, epoch);
                }

            }

        }
        return s_dwsss;
    }


    /**
     * 二阶矩估计
     * s_db = [beta*s_db + (1-beta)*db^2]/(1-beta^epoch)
     * @param s_dbss
     * @param dbss
     * @param beta
     * @param epoch
     * @return
     */
    public List<double[]> getCorrectedSecondRawMomentEstimateOfDbss(List<double[]> s_dbss, List<double[]> dbss, double beta, int epoch){
        for(int i=0; i<s_dbss.size(); i++){
            double[] s_dbs = s_dbss.get(i);
            double[] dbs = dbss.get(i);

            for(int j=0; j<s_dbs.length; j++){
                double s_db = s_dbs[j];
                double db = dbs[j];
                s_dbs[j] = this.getCorrectedSecondRawMoment(s_db, db, beta, epoch);
            }

        }
        return s_dbss;
    }

    /**
     * 一阶矩
     * @param v
     * @param theta
     * @param beta
     * @param epoch
     * @return
     */
    public double getCorrectedFirstMoment(double v, double theta, double beta, int epoch){
        return (beta*v+(1-beta)*theta)/(1-Math.pow(beta, epoch));
    }

    /**
     * 二阶矩
     * @param s
     * @param theta
     * @param beta
     * @param epoch
     * @return
     */
    public double getCorrectedSecondRawMoment(double s, double theta, double beta, int epoch){
        double cs =  (beta*s+(1-beta)*Math.pow(theta, 2)) / (1-Math.pow(beta, epoch));
        return cs;
    }

    /**
     * 更新权重
     * w = w - learningRate * (  v_dw / sqrt(s_dw+epsilon) )
     * @param weightsss
     * @param correctedVDwsss
     * @param correctedSDwsss
     * @param learningRate
     * @param epsilon
     * @return
     */
    public List<double[][]> updateWeightsss(List<double[][]> weightsss, List<double[][]> correctedVDwsss, List<double[][]> correctedSDwsss, double learningRate, double epsilon){
        for(int i=0; i<weightsss.size(); i++){
            double[][] weightss = weightsss.get(i);
            double[][] correctedVDwss = correctedVDwsss.get(i);
            double[][] correctedSDwss = correctedSDwsss.get(i);

            for(int j=0; j<weightss.length; j++){
                double[] weights = weightss[j];
                double[] correctedVDws = correctedVDwss[j];
                double[] correctedSDws = correctedSDwss[j];

                for(int k=0; k<weights.length;  k++){
                    weights[k] = this.updateParam(weights[k], correctedVDws[k], correctedSDws[k], learningRate, epsilon);
                }

            }

        }
        return weightsss;
    }

    /**
     * 更新偏差
     * b = b-learningRate * ( v_db / sqrt(s_db+epsilon) )
     * @param biasss
     * @param correctedVDbss
     * @param correctedSDbss
     * @param learningRate
     * @param epsilon
     * @return
     */
    public List<double[]> updateBiasss(List<double[]> biasss, List<double[]> correctedVDbss, List<double[]> correctedSDbss, double learningRate, double epsilon){
        for(int i=0; i<biasss.size(); i++){
            double[] biass = biasss.get(i);
            double[] correctedVDbs = correctedVDbss.get(i);
            double[] correctedSDbs = correctedSDbss.get(i);

            for(int j=0; j<biass.length; j++){
                biass[j] = this.updateParam(biass[j], correctedVDbs[j], correctedSDbs[j], learningRate, epsilon);
            }

        }
        return biasss;
    }

    public double updateParam(double param, double v, double s, double learningRate, double epsilon){
        return param - learningRate*(v/Math.sqrt(s+epsilon));
    }

    /**
     * 利用学习速率衰减算法自动调节学习速率
     * @param decayRate
     * @param epochNum
     * @return
     */
    public double getDecayedLearningRate(double learningRate0, double decayRate, double epochNum){
        return learningRate0/(1+decayRate*epochNum);
    }

}
