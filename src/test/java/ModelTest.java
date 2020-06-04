import neuralNetwork.Optimizer;
import neuralNetwork.Model;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelTest {

    @Test
    public void testInitialWeights(){
        int[] layerNum = {2, 20, 2};
        Model model = new Model(layerNum);
        double[] exmaple = {2, 1};
        double[] label = {0,1};
        model.forwardProp(exmaple);
        model.backProp(label);
        System.out.println();
    }

    @Test
    public void testOptimizer(){
        int[] shape = {3, 2, 2};
        Model model = new Model(shape);

        List<double[][]> weightsss = new ArrayList<>();
        double[][] weights1 = {{1, -1, 1}, {1, 1, 1}};
//        double[][] weights2 = {{1, 1}, {-1, -1}};
//        weightsss.add(weights1);
//        weightsss.add(weights2);
        //model.setWeights(weightsss);
//        List<double[]> bias = new ArrayList<>();
//        double[] bias1 = {0, 0};
//        double[] bias2 = {0, 0};
//        bias.add(bias1);
//        bias.add(bias2);
        //model.setBiasss(bias);
        double[][] exmaples = {
                 {0, 0, 1},
                {0, 1, 0}, {0, 1, 1},
                {1, 0, 0}, {1, 0, 1},
                {1, 1, 0}, {1, 1, 1}
        };
        double[][] labels = { {0,1}, {1,0},{1,0}, {0,1}, {0,1}, {1,0}, {1,0}};
        Optimizer optimizer = new Optimizer(model);
        model = optimizer.Adam(exmaples, labels, 10000,7, 0.1,1);
        List<double[]> biasss = model.getBiasss();
        System.out.println("*****");
        for(double[] biass : biasss){
            System.out.println(Arrays.toString(biass));
        }
        System.out.println("********");
        for(double[] exmaple : exmaples){
            System.out.println(Arrays.toString(model.predict(exmaple)));
        }
    }

    @Test
    public void testArrays(){
        double[][] a = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
        double[][] b = Arrays.copyOfRange(a, 1 ,a.length);
        for (double[] temp: b){
            System.out.println(Arrays.toString(temp));
        }
    }

    @Test
    public void testZeros(){
        int[] shape = {2, 3, 2};
        Optimizer optimizer = new Optimizer(new Model(shape));
        List<double[][]> zerosss = optimizer.getDwsssWithZero(shape);
        for(double[][] zeross : zerosss) {
            System.out.println("***************");
            for (double[] zeros : zeross) {
                System.out.println(Arrays.toString(zeros));
            }
        }
    }

    @Test
    public void testNaN(){
        System.out.println(Math.pow(0.000000000000000000000000001, -0.00000001000000001));
    }

    @Test
    public void testSaveAndReadModel(){
//        int[] layers = {10, 5, 2};
//        Model model = new Model(layers);
//        NumberRecognition nr = new NumberRecognition();
//        nr.readModelFromLocal();
    }

}
