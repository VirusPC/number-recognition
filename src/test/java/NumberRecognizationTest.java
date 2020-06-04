import numberRecognition.NumberRecognition;
import org.junit.Before;
import org.junit.Test;

public class NumberRecognizationTest {
    NumberRecognition nr;

    @Before
    public void before(){
//        nr = new NumberRecognition();
        nr = null;
    }

    @Test
    public void testTrain(){
//        nr.train();
//        nr.writeModelToLocal();
//        System.out.println("train error: " + nr.getTrainError());
//        System.out.println("test error: "+nr.getTestError());
    }

    @Test
    public void testPredict(){
       String imgPath = "src/resource/testSet/0/";
       String result = nr.predict(imgPath);
       System.out.println("预测结果为："+result);
    }
}
