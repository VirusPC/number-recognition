package img;

import com.sun.imageio.plugins.common.ImageUtil;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;

public class ImageProcessor {

    private static final int CHANNEL_NUM = 3;
    private static final int RED_CHANNEL = 0;
    private static final int GREEN_CHANNEL = 1;
    private static final int BLUE_CHANNEL = 2;

    // 背景色为白
    private static final int backgroundColor = 0xff;

    // 前景色为黑
    private static final int foregroundColor = 0;

    // sigma=0.8的高斯滤波器，sigma应随模板增大而增大。
    private static final double[][] GAUSS_FILTER = {
            {0.0997, 0.1163, 0.0997},
            {0.1163, 0.1358, 0.1163},
            {0.0997, 0.1163, 0.0997}
    };

    // 拉普拉斯滤波器
    private static final double[][] LAPLACE_FILTER = {
            {1, 4, 1},
            {4, -20, 4},
            {1, 4, 1}
    };

    // 膨胀滤波器
    private static final double[][] EXPAND_FILTER = {
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
    };

//    private static final int ALPHA_MASK = 0xff000000;
//    private static final int RED_MASK = 0x00ff0000;
//    private static final int GREEM_MASK = 0x0000ff00;
//    private static final int BLUE_MASK = 0x000000ff;
//
//    private static final int ALPHA_OPACITY = 0xff;
//
//    private double[][][] rgbImg;
//    private int width;
//    private int height;
//    private int imgType;
//    private int formate;


    private BufferedImage img;


    public static void main(String[] args) throws Exception {

        File srcDir = new File("src/resource/dataSet");
        File dstDir = new File("src/resource/dataSetSharp30by30s");
        for (File srcDirForOne : srcDir.listFiles()) {
            File dstDirForOne = new File(dstDir.getPath() + "/" + srcDirForOne.getName());
            if (!dstDirForOne.exists()) {
                dstDirForOne.mkdirs();
            }
            for (File srcFile : srcDirForOne.listFiles()) {
                File dstFile = new File(dstDirForOne.getPath() + "/" + srcFile.getName());
                if (!dstFile.exists()) {
                    dstFile.createNewFile();
                }

                ImageProcessor imgProcessor = new ImageProcessor(srcFile);
                imgProcessor.rgb2Gray();

                imgProcessor.nearestNeighborInterpolationScale(100, 100);
                imgProcessor.gaussBlur();
                imgProcessor.OSTUBinarize();
                imgProcessor.selfAdaptionDilateAndErrode();
                imgProcessor.segment();
                imgProcessor.nearestNeighborInterpolationScale(30, 30);
                imgProcessor.laplacianSharpening();

                ImageIO.write(imgProcessor.getImage(), "png", dstFile);
                //ImageIO.getReaderFormatNames();
            }
        }

    }

    public ImageProcessor(BufferedImage img){
        this.img = img;
    }

    public ImageProcessor(File imgFile){
        getImageByFile(imgFile);
    }

    public ImageProcessor(String path){
        readImageFromLocal(path);
    }


    private void readImageFromLocal(String path){
        File imgFile = new File(path);
        if(!imgFile.exists() || !imgFile.isFile()){
            System.err.println("image file not exists");
            img = null;
        } else {
            getImageByFile(imgFile);
        }
    }

    private void getImageByFile(File imgFile){

        try{
            img = ImageIO.read(imgFile);
        } catch (Exception e){
            img = null;
            System.err.println("read image error");
        }
    }

    public boolean saveImage(String path, String name, String format){
        File imgDir = new File(path);
        if(!imgDir.exists() || !imgDir.isDirectory()){
            imgDir.mkdirs();
        }
        File imgFile = new File(path+"\\" +name);

        try{
            ImageIO.write(img, format, imgFile );
        } catch(Exception e){
            System.err.println("image write error");
            return false;
        }
        return true;
    }


    public BufferedImage getImage(){
        if(img==null){
            System.out.println("不存在图片");
        }
        return img;
    }

    public double[] preprocess(){

        double[] normlizedImg = this.normalize();
        return normlizedImg;
    }

    private int getAlpha(int pixel){
        return (pixel>>24) & 0xff;
    }

    private int getRed(int pixel){
        return (pixel>>16) & 0xff;
    }

    private int getGreen(int pixel){
        return  (pixel >>8) & 0xff;
    }

    private int getBlue(int pixel){
        return pixel & 0xff;
    }

    private int getGray(int pixel){
        return pixel & 0xff;
    }

    private int getRGBPixel(int alpha, int r, int g, int b){
        return (alpha<<24) | (r<<16) | (g<<8) | b;
    }

    private int getGrayPixel(int alpha, int gray){
        return (alpha<<24)|(gray<<16) | (gray<<8) | gray;
    }


    /**
     * 灰度化
     * @return
     */
    public void rgb2Gray() {
        // 获取高度和宽度
        int width = img.getWidth();
        int height = img.getHeight();

        // 对每个像素点进行转换
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = img.getRGB(i, j);
                int r = getRed(pixel);
                int g = getGreen(pixel);
                int b = getBlue(pixel);
                int alpha = getAlpha(pixel);
                int gray = (r*38 + g*75 + b*15) >> 7;
                // alpha通道，调整透明度
                int newPixel = getGrayPixel(alpha, gray);
                img.setRGB(i, j, newPixel);
            }
        }
    }

    /**
     * 固定阈值二值化
     *
     * @return
     */
    public void binarize(int threshhold) {
        int black = 0xff000000;
        int white = 0xffffffff;
        threshhold = threshhold | (threshhold << 8) | (threshhold << 16) | (255 << 24);
        int width = img.getWidth();
        int height = img.getHeight();

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int color = img.getRGB(i, j);
                if (color < threshhold) {
                    color = black;
                } else {
                    color = white;
                }
                img.setRGB(i, j, color);
            }
        }
    }

    /**
     * OSTU二值化(大津算法)
     *
     * @return
     */
    public void OSTUBinarize() {
        int width = img.getWidth();
        int height = img.getHeight();
        double totalPixelCount = width * height;

        // 直方图
        double[] histogram = new double[256];

        // 生成直方图
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int pixel = img.getRGB(x, y);
                int gray = getGray(pixel);
                histogram[gray] += 1;
            }
        }

        // 归一化直方图
        for (int i = 0; i < 256; i++) {
            histogram[i] /= totalPixelCount;
        }

        // 保存最大方差及对应阈值
        double variance = 0;
        int threshold = 0;

        // 阈值从0开始迭代
        for (int itrThreshold = 1; itrThreshold < 256; itrThreshold++) {
            // 计算阈值以下的像素所占比例w0及平均灰度u0
            double darkPixelCount = 0;
            double darkGrayCount = 0;
            for (int gray = 0; gray < itrThreshold; gray++) {
                darkPixelCount += histogram[gray];
                darkGrayCount += histogram[gray] * gray;
            }
            double w0 = darkPixelCount / totalPixelCount;
            double u0 = darkGrayCount / darkPixelCount;

            // 计算阈值以下的像素所占比例w1及平均灰度u1
            double lightPixelCount = 0;
            double lightGrayCount = 0;
            for (int gray = itrThreshold; gray < 256; gray++) {
                lightPixelCount += histogram[gray];
                lightGrayCount += histogram[gray] * gray;
            }
            double w1 = lightPixelCount / totalPixelCount;
            double u1 = lightGrayCount / lightPixelCount;

            double itrVariance = w0 * w1 * (u0 - u1) * (u0 - u1);
            if (itrVariance > variance) {
                variance = itrVariance;
                threshold = itrThreshold;
            }
        }

        binarize(threshold);
    }

    /**
     * 反色
     * @return
     */
    private void inverseColor(){
        int width = img.getWidth();
        int height = img.getHeight();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int pixel = img.getRGB(x, y);
                //System.out.println(pixel);
                int r = getRed(pixel);
                int g = getGreen(pixel);
                int b = getBlue(pixel);
                int alpha = getAlpha(pixel);
                r = 0xff-r;
                g = 0xff-g;
                b = 0xff-b;
                pixel = getRGBPixel(alpha, r, g, b);
                img.setRGB(x, y, pixel);
            }
        }
    }

    /**
     * 获取黑色像素占总像素的比例
     * @return
     */
    public double getPropotionOfBlack(){

        int width  = img.getWidth();
        int height = img.getHeight();

        int whiteCount = 0;
        int blackCount = 0;

        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int pixel = img.getRGB(x, y);
                int gray = getGray(pixel);
                //System.out.println(gray);
                if(gray == 0){
                    blackCount++;
                }
            }
        }

        double portion = blackCount/(width*height);
        return portion;
    }

    /**
     * 用于灰度图反色
     * 设置前景为黑色
     * 假设数字所占面积小于背景所占面积
     * @return
     */
    public void setForegroundToBlack(){
        if(getPropotionOfBlack()<0.5){
            inverseColor();
        }
    }



    /**
     * 使用 最近邻插值 进行图片放缩
     *
     * @param targetWidth  要放缩到的图片宽度
     * @param targetHeight 要放缩到的图片高度
     * @return
     */
    public void nearestNeighborInterpolationScale(int targetWidth, int targetHeight) {
        BufferedImage scaledImg = new BufferedImage(targetWidth, targetHeight, img.getType());

        int width = img.getWidth();
        int height = img.getHeight();

        double xScale = (double) width / (double) targetWidth;
        double yScale = (double) height / (double) targetHeight;

        for (int targetX = 0; targetX < targetWidth; targetX++) {
            for (int targetY = 0; targetY < targetHeight; targetY++) {
                double x = targetX * xScale;
                double y = targetY * yScale;
                int pixel = nearestNeighborInterpolation(x, y);
                scaledImg.setRGB(targetX, targetY, pixel);
            }
        }
        this.img = scaledImg;
    }

    /**
     * 最邻近插值
     * @param x
     * @param y
     * @return
     */
    private int nearestNeighborInterpolation(double x, double y){
        return img.getRGB((int) Math.min(Math.round(x), img.getWidth() - 1), (int) Math.min(Math.round(y), img.getHeight() - 1));
        //return img.getRGB((int)Math.round(x), (int)Math.round(y));
    }

    public void bilinearInterpolationScale(int targetWidth, int targetHeight) {
        BufferedImage scaledImg = new BufferedImage(targetWidth, targetHeight, img.getType());

        int width = img.getWidth();
        int height = img.getHeight();

        double xScale = (double) width / (double) targetWidth;
        double yScale = (double) height / (double) targetHeight;

        for (int targetX = 0; targetX < targetWidth; targetX++) {
            for (int targetY = 0; targetY < targetHeight; targetY++) {
                double x = targetX * xScale;
                double y = targetY * yScale;
                int pixel = bilinearInterpolation(x, y);
                scaledImg.setRGB(targetX, targetY, pixel);
            }
        }
        this.img = scaledImg;
    }


    /**
     * 双线性插值
     * 进行双线性插值后
     *
     * @param x   映射到原图片上的横坐标
     * @param y   映射到原图片上的纵坐标
     * @return 像素值
     */
    private int bilinearInterpolation(double x, double y) {
        int pixel = 0;
        double epsilon=0.0001;

        // 4个最邻近像素的坐标 (x1, y1)、(x1, y2)、(x2, y1)、(x2, y2)
        int x1 = (int) x;
        int x2 = x1 + 1;
        int y1 = (int) y;
        int y2 = y1 + 1;

        //四个最邻近像素值
        int p1,p2,p3,p4;
        //两个插值中间值
        int p12,p34;

        int width = img.getWidth();
        int height = img.getHeight();

        // 点不在图像范围内时，返回 -1；
        if((x<0)||(x>width-1)||(y<0)||(y>height-1)) {
            //System.out.println("error");
            pixel = nearestNeighborInterpolation(x, y);
        }

        if (Math.abs(x - width + 1) <= epsilon) {
            //如果计算的点在图像右边缘上
            if(Math.abs(y-height+1)<=epsilon){
                //如果计算地点刚好是图像最右下角的那一个像素，直接返回该点像素值
                pixel = getGray(img.getRGB(x1, y1));
            }
            else{
                //如果图像在右边缘上且不是最后一点，直接一次插值
                p1 = getGray(img.getRGB(x1, y1));
                p3 = getGray(img.getRGB(x1, y2));

                pixel = (int)(p1+(y-y1)*(p3-p1));
            }
        } else if(Math.abs(y-height+1)<=epsilon){
            //如果计算地点在图像下边缘上且不是最后一点，直接一次插值即可
            p1 = getGray(img.getRGB(x1, y1));
            p2 = getGray(img.getRGB(x2, y1));
            System.out.println(p1);
            pixel = (int)(p1+(x-x1)*(p2-p1));
        } else {
            //计算四个最邻近像素值
            p1 = getGray(img.getRGB(x1, y1));
            p2 = getGray(img.getRGB(x2, y1));
            p3 = getGray(img.getRGB(x1, y2));
            p4 = getGray(img.getRGB(x2, y2));
            //System.out.println(p1);
            //插值
            p12 = (int)(p1+(x-x1)*(p2-p1));
            p34 = (int)(p3+(x-x1)*(p4-p3));

            pixel = ((int)(p12+(y-y1))*(p34-p12));
        }
        //System.out.println(pixel);
        return getGrayPixel(0xff, pixel);

    }

//    /**
//     * 图像转向量（一维数组）
//     *
//     * @param img
//     * @return
//     */
//    public double[] vectorizeImg(BufferedImage img) {
//        int width = img.getWidth();
//        int height = img.getHeight();
//        int vectorLength = width * height;
//        double[] v = new double[vectorLength];
//
//        for (int x = 0; x < width; x++) {
//            for (int y = 0; y < height; y++) {
//                int pixel = img.getRGB(x, y);
//                int gray = (pixel & 0xff) / 255;
//                v[x + height * width] = gray;
//            }
//        }
//
//        return v;
//    }

//    /**
//     * rgb图转三维数组
//     * @param img
//     * @return
//     */
//    private double[][][] rgbImg2Matrixes(BufferedImage img){
//
//        int width = img.getWidth();
//        int height = img.getHeight();
//        double[][][] rgbMatrixes = new double[width][height][CHANNEL_NUM];
//
//        // 对每个像素点进行转换
//        for (int x = 0; x < width; x++) {
//            for (int y = 0; y < height; y++) {
//                int pixel = img.getRGB(x, y);
//
//                int r = getRed(pixel);
//                int g = getGreen(pixel);
//                int b = getBlue(pixel);
//
//                rgbMatrixes[x][y][RED_CHANNEL] = r;
//                rgbMatrixes[x][y][GREEN_CHANNEL] = g;
//                rgbMatrixes[x][y][BLUE_CHANNEL] = b;
//
//            }
//        }
//
//        return rgbMatrixes;
//    }
//
//    /**
//     * 三维数组转rgb图
//     */
//    public BufferedImage rgbMatrixes2Img(double[][][] rgbMatrixes, int imgType){
//
//        int width = rgbMatrixes.length;
//        int height = rgbMatrixes[0].length;
//
//        BufferedImage img = new BufferedImage(width, height, imgType);
//        for(int x=0; x<width; x++){
//            for(int y=0; y<width; y++){
//                int r = (int)rgbMatrixes[x][y][RED_CHANNEL];
//                int g = (int)rgbMatrixes[x][y][GREEN_CHANNEL];
//                int b = (int)rgbMatrixes[x][y][BLUE_CHANNEL];
//                int pixel = 0xff<<24 | r<<16 | g<<8 | b;
//                img.setRGB(x, y, pixel);
//            }
//        }
//
//        return img;
//    }
//
//    /**
//     * rgb矩阵 转 灰度矩阵
//     * @param rgbMatrixes
//     * @return
//     */
//    public double[][] rgbMatrixes2GrayMatrix(double[][][] rgbMatrixes){
//
//        double redWeight = 0.3;
//        double greenWeight = 0.59;
//        double blueWeight = 0.11;
//
//        double width = rgbMatrixes.length;
//        double height = rgbMatrixes[0].length;
//
//        double[][] grayMatrix = new double[rgbMatrixes.length][];
//
//        for(int x=0; x<width; x++){
//            for(int y=0; y<height; y++){
//                grayMatrix[x][y] = redWeight * rgbMatrixes[x][y][RED_CHANNEL]
//                        + greenWeight * rgbMatrixes[x][y][GREEN_CHANNEL]
//                        + blueWeight * rgbMatrixes[x][y][BLUE_CHANNEL];
//            }
//        }
//
//        return grayMatrix;
//    }
//
//    public double[][][] grayMatrix2rgbMatrixes(double[][] grayMatrix){
//
//        int width = grayMatrix.length;
//        int height = grayMatrix[0].length;
//
//        double[][][] rgbMatrixes = new double[width][height][CHANNEL_NUM];
//
//        for(int x=0; x<width; x++){
//            for(int y=0; y<height; y++){
//                double gray = grayMatrix[x][y];
//                rgbMatrixes[x][y][RED_CHANNEL] = gray;
//                rgbMatrixes[x][y][GREEN_CHANNEL] = gray;
//                rgbMatrixes[x][y][BLUE_CHANNEL] = gray;
//            }
//        }
//        return rgbMatrixes;
//    }


    /**
     * 相关
     * 与卷积相对应
     * stride 为1
     * filter边长需为奇数
     * @param filter 边长为奇数的过滤器
     * @return
     */
    private void correlationWithSamePadding(double[][] filter){
        //if(filter.length%2!=1)
        BufferedImage img2 = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        img2.setData(img.getData());

        int width = img.getWidth();
        int height = img.getHeight();

        int filterWidth = filter.length;
        int filterHeight = filter[0].length;

//        /*
//         宽度（高度）为奇数时，左右（上下）paddingSize 相等。
//         宽度（高度）为偶数时，左（上）侧 paddingSize 比 右（下）侧 padingSize 大 1
//          */
        int leftPaddingSize = (filterWidth-1)/2;
        int upperPaddingSize = (filterHeight-1)/2;

        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int newGray = 0;

                for(int fx = 0; fx<filterWidth; fx++){
                    for(int fy = 0; fy<filterHeight; fy++){
                        int[] position = getNearestGrayPosition(x+fx-leftPaddingSize, y+fy-upperPaddingSize, width, height);
                        int pixel = img.getRGB(position[0], position[1]);
                        int gray = getGray(pixel);
                         newGray += gray*filter[fx][fy];
                    }
                }

                // 防止超出像素范围
                if(newGray>0xff){
                    newGray = 0xff;
                } else if(newGray<0){
                    newGray = 0;
                }
                img2.setRGB(x, y, getGrayPixel(0xff, newGray));

            }
        }

        img = img2;
    }

    /**
     * 获取与该位置最近的，位于图片内的位置
     * @param x
     * @param y
     * @param width
     * @param height
     * @return
     */
    private int[] getNearestGrayPosition(int x, int y, int width, int height){
        int[] position = new int[2];
        if(x<0){
            x = 0;
        } else if(x>=width){
            x = width-1;
        }

        if(y<0){
            y=0;
        } else if(y>=height){
            y = height-1;
        }

        position[0] = x;
        position[1] = y;

        return position;
    }

//    /**
//     * 填充虚拟边界的内容总是重复与它最近的边缘像素
//     * @param img
//     * @param paddingSize
//     * @return
//     */
//    public BufferedImage replicatePadding(BufferedImage img, int paddingSize){
//        return img;
//    }

    /**
     * 灰度图高斯平滑（模糊）
     * @return
     */
    public void gaussBlur(){
        correlationWithSamePadding(GAUSS_FILTER);
    }

    /**
     * 拉普拉斯锐化
     * @return
     */
    public void laplacianSharpening(){
        correlationWithSamePadding(LAPLACE_FILTER);
    }


    /**
     * 用 fWidth * fHeight 的正方形结构元素进行膨胀运算
     * @return
     */
    public void dilate(int fHeight, int fWidth){
        if(fHeight*fWidth<=1){
            return;
        }
        BufferedImage img2 = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        img2.setData(img.getData());

        int width = img.getWidth();
        int height = img.getHeight();

        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int newGray = 255;

                for(int fx = 0; fx<fWidth; fx++){

                    for(int fy = 0; fy<fHeight; fy++){
                        int[] position = getNearestGrayPosition(x+fx-1, y+fy-1, width, height);
                        int pixel = img.getRGB(position[0], position[1]);
                        int gray = getGray(pixel);
                        if(gray==0){
                            newGray = 0;
                            break;
                        }
                    }
                    if(newGray==0){
                        break;
                    }
                }

                img2.setRGB(x, y, getGrayPixel(0xff, newGray));

            }
        }

        img = img2;
    }

    /**
     * 腐蚀运算
     * @param fHeight
     * @param fWidth
     * @return
     */
    public void errode(int fHeight, int fWidth){

        if(fHeight*fWidth<=1){
            return;
        }

        BufferedImage img2 = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        img2.setData(img.getData());

        int width = img.getWidth();
        int height = img.getHeight();

        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int newGray = 0;

                for(int fx = 0; fx<fWidth; fx++){

                    for(int fy = 0; fy<fHeight; fy++){
                        int[] position = getNearestGrayPosition(x+fx-1, y+fy-1, width, height);
                        int pixel = img.getRGB(position[0], position[1]);
                        int gray = getGray(pixel);
                        if(gray==255){
                            newGray = 255;
                            break;
                        }
                    }
                    if(newGray==255){
                        break;
                    }
                }

                img2.setRGB(x, y, getGrayPixel(0xff, newGray));

            }
        }

        img = img2;
    }

    public void selfAdaptionDilateAndErrode(){
        double dilateThreshold = 0.3;
        double errodeThreshold = 0.7;

        double proportionOfBlack = getPropotionOfBlack();

        if(proportionOfBlack<errodeThreshold && proportionOfBlack>dilateThreshold){
            return;
        }

        int fHeight = 3;
        int fWidth = 3;

        if(proportionOfBlack<dilateThreshold){
            dilate(fHeight, fWidth);
        }else if(proportionOfBlack>errodeThreshold){
            errode(fHeight, fWidth);
        }

    }


    /**
     * 细化
     * @param grayImg
     * @return
     */
    public BufferedImage thining(BufferedImage grayImg){

        int width = grayImg.getWidth();
        int height = grayImg.getHeight();

        // 2 <= NZ(P1) <=6
        boolean condition1 = false;
        // Z0(P1) = 1
        boolean condition2 = false;
        // (P2 * P4 * P8 == 0) OR (Z0(P1) != 1)
        boolean condition3 = false;
        // (P2 * P4 *P6 == 0) OR (Z0(P4) !=1)
        boolean condition4 = false;
        // 图片是否被改变
        boolean modified = true;



        return grayImg;
    }

    /**
     * 字符边缘切割
     */
    public void segment(){

        int width = img.getWidth();
        int height = img.getHeight();

        BufferedImage img2 = img.getSubimage(0, 0, width, height);

        laplacianSharpening();

        int top,bottom,left,right=0;
        top = getTop();
        bottom = getBottom();
        left = getLeft();
        right = getRight();

        img = img2.getSubimage(left, top, Math.min(right-left+2, width-left), Math.min(bottom-top+2, height-top));

    }

    public int getLeft(){
        int left = 0;
        int width = img.getWidth();
        int height = img.getHeight();
        boolean over = false;
        // 从左往右扫描
        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int gray = getGray(img.getRGB(x, y));
                if(gray>128){
                    left = x;
                    over = true;
                    break;
                }
            }
            if(over==true){
                break;
            }
        }
        return left;
    }

    public int getRight(){
        int right = 0;
        int width = img.getWidth();
        int height = img.getHeight();
        boolean over = false;
        //从右往左扫描
        for(int x=width-1; x>0; x--){
            for(int y=0; y<height; y++){
                int gray = getGray(img.getRGB(x, y));
                if(gray>128){
                    right = x;
                    over = true;
                    break;
                }
            }
            if(over==true){
                break;
            }
        }
        return right;
    }

    public int getTop(){
        int top = 0;
        int width = img.getWidth();
        int height = img.getHeight();
        boolean over = false;
        // 从上往下扫描
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                int gray = getGray(img.getRGB(x, y));
                if(gray>128){
                    top = y;
                    over = true;
                    break;
                }
            }
            if(over==true){
                break;
            }
        }
        return top;
    }

    public int getBottom(){
        int bottom = 0;
        int width = img.getWidth();
        int height = img.getHeight();
        boolean over = false;
        // 从下往上扫描
        for(int y=height-1; y>0; y--){
            for(int x=0; x<width; x++){
                int gray = getGray(img.getRGB(x, y));
                if(gray>128){
                    bottom = y;
                    over = true;
                    break;
                }
            }
            if(over==true){
                break;
            }
        }
        return bottom;
    }



    /**
     * 对灰度图进行归一化
     * @return
     */
    public double[] normalize(){
        int width = img.getWidth();
        int height = img.getHeight();

        double[] normImg = new double[width*height];

        for(int x=0; x<width; x++){
            for(int y=0; y<height; y++){
                int gray = getGray(img.getRGB(x, y));
                normImg[height*y + x] = gray>125?0.5:-0.5;
            }
        }

        return normImg;
    }


}