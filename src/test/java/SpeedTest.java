import neuralNetwork.Layer;
import neuralNetwork.Model;
import neuralNetwork.Neron;
import neuralNetwork.ReluLayer;
import org.junit.Test;

public class SpeedTest {
    @Test
    public void testGetSet(){
        Model model = new Model();

        for(int i=0; i<=1000; i++){
            Layer layer = new ReluLayer();
            for(int j=0; j<=1000; j++){
                Neron neron = new Neron();
                layer.addNeron(neron);
            }
            model.addLayer(layer);
        }

//        double start = System.currentTimeMillis();
//        for(int count=0; count<=100; count++) {
//            for (int i = 0; i <= 1000; i++) {
//                Layer layer = model.getLayer(i);
//                for (int j = 0; j <= 1000; j++) {
//                    Neron neron = layer.getNeron(j);
//                    neron.setZR(3);
//                }
//            }
//        }
//
//        double end = System.currentTimeMillis();
//        double pass =end-start;

        double start = System.currentTimeMillis();
        for(int count=0; count<=1000; count++) {
            for (int i = 0; i <= 1000; i++) {
                Layer layer = model.getLayer(i);
                for (int j = 0; j <= 1000; j++) {
                    //Neron neron = layer.getNeron(j);
                    //neron.setZ(30);
                    //neron.z=
                }
            }
        }

        double end = System.currentTimeMillis();
        double pass2 = end-start;
        //System.out.println("time1:"+pass);
        System.out.println("time2:"+pass2);
    }
}
