//package img;
//
//import org.opencv.core.Core;
//import org.opencv.core.CvType;
//import org.opencv.core.Mat;
//import org.opencv.core.Rect;
//import org.opencv.imgcodecs.Imgcodecs;
//import org.opencv.imgproc.Imgproc;
//
//import java.io.File;
//import java.util.Arrays;
//
//public class ImgPreprocessOpenCV {
//
//    public static void main(String[] args){
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        //File file = new File("src/resource/Bmp/Sample001/img001-00001.png");
//        //System.out.println(file.exists());
//        Mat img= Imgcodecs.imread("src/resource/Bmp/Sample001/img001-00001.png", Imgcodecs.IMREAD_GRAYSCALE);
//        //img2 = Imgproc.
//        //double[]
//        //Rect r = new Rect();
//        Mat img2 =img.clone() ;
//
//        //for
//
//        Imgcodecs.imwrite("src/resource/processed/img001-00001.png", img2);
       // System.out.println(Arrays.toString(d));

        //System.out.println(a);
        //System.out.println(img);
        //Mat end=new Mat();



//        //去除干扰线
//
//        for(int x=0;x<img.rows();x++){
//            for(int y=0;y<img.cols();y++){
//                double[] clone=img.get(x, y).clone();   //修改通道数值
//                double cb=clone[0];
//                double cg=clone[1];
//                double cr=clone[2];
//                double avg=(cb+cg+cr)/3;
//                if(!((cb>c&&cg>c&&cr>c)&&((avg>98)&&(avg<148)||(avg>153)&&(avg<196))&&((max(clone)-min(clone))<55))){
//                    clone[0]=255;
//                    clone[1]=254;
//                    clone[2]=255;
//                    img.put(x, y, clone);
//                }
//            }
//        }
//
//
//        Imgproc.bilateralFilter(img, end, a, a*2, a/2);  //双边滤波
//
//    }
//}
