package numberRecognition;

import img.ImageProcessor;
import neuralNetwork.Model;
import neuralNetwork.Optimizer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NumberRecognition {

//        String[] order = {
//                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
//                "aLower", "bLower", "cLower", "dLower", "eLower", "fLower", "gLower",
//                "hLower", "iLower", "jLower", "kLower", "lLower", "mLower", "nLower",
//                "oLower", "pLower", "qLower", "rLower", "sLower", "tLower",
//                "uLower", "vLower", "wLower", "xLower", "yLower", "zLower",
//                "A", "B", "C", "D", "E", "F", "G",
//                "H", "I", "J", "K", "L", "M", "N",
//                "O", "P", "Q", "R", "S", "T",
//                "U", "V", "W", "X", "Y", "Z"
//        };

    String[] order = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    };

    Model model = null;
    Optimizer optimizer = null;
    ImageProcessor imageProcessor = null;


    public static void main(String[] args){

        int[] shape = {30*30, 50, 10};

        String modelReadPath = "src/models";
        String modelSavePath = modelReadPath;
        String modelName = "model_30by30_50_0.001_fdaf";

        String trainSetPath="src/resource/trainSet430by30s";
        String testSetPath="src/resource/testSet430by30s";

        int itr = 50;
        int batchSize = 1000;
        double learningRate = 0.005;//0.0001
        double lambda2 = 0.001;

        NumberRecognition nr = new NumberRecognition(modelReadPath, modelName, shape);

        String imgPath = "src/resource/dataSet/9/20.png";
        String predictLabel = nr.predict(imgPath);
        System.out.println("预测结果为: " + predictLabel);

        nr.train(trainSetPath, itr, batchSize, learningRate, lambda2);

        nr.writeModelToLocal(modelSavePath, modelName);
        System.out.println("train error: " + nr.getError(trainSetPath));
        System.out.println("test error: "+nr.getError(testSetPath));
    }

    public NumberRecognition(String modelReadPath, String modelName, int[] shape){
        model = readModelFromLocal(modelReadPath, modelName);
        if(model==null){
            model = new Model(shape);
        }
    }


    /**
     * 预测图片类别
     * @param imgPath
     * @return
     */
    public String predict(String imgPath) {
        //Model model = readModelFromLocal();
        double[] example = getPreprocessedAndNormalizedImgFromLocal(imgPath);
        int orderNum = model.getPredictedClassOrder(example);
        return order[orderNum];
    }



    private Model readModelFromLocal(String modelReadPath, String modelName){
        File file = new File(modelReadPath + "\\" + modelName );
        if(!file.exists()){
            return null;
        }

        try {
            ObjectInputStream ois = new ObjectInputStream(
                    new FileInputStream(
                            file
                    )
            );
            Object model = ois.readObject();
            return (Model)model;
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("model read error");
        }
        return null;
    }


    public void writeModelToLocal(String modelSavePath,String modelName){
        ObjectOutputStream oos = null;
        try {
            File dir = new File(modelSavePath);
            if(!dir.exists()){
                dir.mkdirs();
            }
            File saveFile = new File(modelSavePath + "\\" + modelName);

            oos = new ObjectOutputStream(
                    new FileOutputStream(
                            saveFile
                    )
            );
            oos.writeObject(model);
            oos.close();
            System.out.println("已保存");
        } catch(Exception e){
            e.printStackTrace();
            System.out.println("write error");
        }
    }


    private double[] getPreprocessedAndNormalizedImgFromLocal(String path){
        File imgFile = new File(path);
        if(!imgFile.exists()){
            System.out.println("read img error");
            return null;
        }
        try {
            BufferedImage img = ImageIO.read(imgFile);
            imageProcessor = new ImageProcessor(img);
            imageProcessor.rgb2Gray();
            imageProcessor.nearestNeighborInterpolationScale(100, 100);
            imageProcessor.gaussBlur();
            imageProcessor.OSTUBinarize();
            imageProcessor.selfAdaptionDilateAndErrode();
            imageProcessor.segment();
            imageProcessor.nearestNeighborInterpolationScale(30, 30);
            imageProcessor.laplacianSharpening();
            double[] normedImg = imageProcessor.normalize();
            return normedImg;
        } catch (Exception e){
            System.out.println("error");
            return null;
        }
    }



    public void train(String trainSetPath, int itr, int batchSize, double learningRate, double lambda2){

        List<double[][]> examplesAndLabels = null;
        double[][] examples = null;
        double[][] labels = null;
        try {
            examplesAndLabels = getExamplesAndLabelsFromLocal(trainSetPath, order);
            examples = examplesAndLabels.get(0);
            labels  = examplesAndLabels.get(1);
        } catch(Exception e){
            System.out.println("read examples error");
        }
        optimizer = new Optimizer(model);
        model = optimizer.Adam(examples, labels,itr, batchSize, learningRate, lambda2);
    }


    /**
     * 获取错误率
     * @param path
     * @return
     */
    public double getError(String path){
        List<double[][]> examplesAndLabels = null;
        double[][] examples = null;
        double[][] labels = null;
        try {
            examplesAndLabels = getExamplesAndLabelsFromLocal(path, order);
            examples = examplesAndLabels.get(0);
            labels  = examplesAndLabels.get(1);
        } catch(Exception e) {
            System.out.println("read train or test set error");
        }
        double error = model.getError(examples, labels);
        return error;
    }


    /**
     * 获取归一化后的样本及其标签
     * 图片所在的文件夹的名称即为标签
     * 每个标签下的图片个数需相同
     * @param rootPath
     * @param order label与列表序号的对应关系
     */
    private List<double[][]> getExamplesAndLabelsFromLocal(String rootPath, String[] order) throws Exception{
        List<double[][]> examplesAndLabels = new ArrayList<>();

        // 有待改进
        int[] imgNumEachLabel = new int[order.length];
        for(int labelOrder=0; labelOrder<order.length; labelOrder++){
            String labelName = order[labelOrder];
            File labelFile = new File(rootPath+"/"+labelName);
            if(!labelFile.exists()){
                System.out.println("file read error");
            }
            imgNumEachLabel[labelOrder] = labelFile.list().length;
        }

        // 保证各个数字的样本数量相同
        int minNumEachLabel = imgNumEachLabel[0];
        for(int i=1; i<imgNumEachLabel.length; i++){
            if(imgNumEachLabel[i]<minNumEachLabel){
                minNumEachLabel = imgNumEachLabel[i];
            }
        }

        int totalNum = minNumEachLabel*order.length;

        double[][] examples = new double[totalNum][];
        double[][] labels = new double[totalNum][];

        // 获取本地图片并归一化
        //int count=0;
        for(int labelOrder=0; labelOrder<order.length; labelOrder++) {
            String labelName = order[labelOrder];
            File labelFile = new File(rootPath+"/"+labelName);
            if(!labelFile.exists()){
                System.out.println("file read error");
            }

            File[] imgFilesWithSameLabel = labelFile.listFiles();

            for (int i = 0; i < minNumEachLabel; i++) {
                BufferedImage img = ImageIO.read(imgFilesWithSameLabel[i]);
                imageProcessor = new ImageProcessor(img);
                double[] normalizedImg = imageProcessor.normalize();
                //examples[i*order.length+labelOrder] = normalizedImg;
                int pos = i*order.length+labelOrder;
                double[] label = new double[order.length];
                label[labelOrder] = 1;
                examples[pos] = normalizedImg;
                labels[pos] = label;
                //count++;
            }
        }

        examplesAndLabels.add(examples);
        examplesAndLabels.add(labels);
        return examplesAndLabels;
    }

    public Model getModel(){
        return model;
    }

}
