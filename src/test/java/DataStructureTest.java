import neuralNetwork.Layer;
import neuralNetwork.ReluLayer;
import org.junit.Test;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class DataStructureTest {
    @Test
    public void testLinkedList(){
        Layer layer1 = new ReluLayer();
        Layer layer2 = new ReluLayer();
        Layer layer3 = new ReluLayer();
        List<Layer> layers = new LinkedList<Layer>();
        layers.add(layer1);
        layers.add(layer2);
        layers.add(layer3);
        Layer l = layers.get(1);
        Map map = new HashMap<>();
    }

    @Test
    public void testDouble2Int(){
        double a = 1.9;
        System.out.println((int)a);
    }
}
