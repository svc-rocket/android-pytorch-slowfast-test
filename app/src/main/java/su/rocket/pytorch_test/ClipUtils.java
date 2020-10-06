package su.rocket.pytorch_test;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_32FC3;

public class ClipUtils {
    private static String TAG = "ImageUtils";

    public static Tensor[] prepare(List<Mat> clip) {
        INDArray fastIndexes = Nd4j.linspace(0, clip.size() - 1, 32, DataType.INT);
        INDArray slowIndexes = Nd4j.linspace(0, clip.size() - 1, 8, DataType.INT);

        int[] fastIndexesArray = fastIndexes.toIntVector();
        int[] slowIndexesArray = slowIndexes.toIntVector();

        float[] fastArray = new float[0];
        float[] slowArray = new float[0];
        for (int i = 0; i < clip.size(); i++) {
            Mat mat = clip.get(i);
            if (ArrayUtils.contains(fastIndexesArray, i) ||
                ArrayUtils.contains(slowIndexesArray, i)) {

                Mat scaled = scale(mat);

                Mat converted = new Mat();
                Imgproc.cvtColor(scaled, converted, Imgproc.COLOR_RGBA2RGB);
                scaled.release();

                MatOfFloat floatMat = new MatOfFloat();
                converted.convertTo(floatMat, CV_32FC3, 1.0 / 255);
                converted.release();

                long size = floatMat.total();
                int channels = floatMat.channels();
                float[] floatArray = new float[(int) (size * channels)];
                floatMat.get(0, 0, floatArray);
                floatMat.release();

                if (ArrayUtils.contains(fastIndexesArray, i)) {
                    fastArray = concat(fastArray, floatArray);
                }

                if (ArrayUtils.contains(slowIndexesArray, i)) {
                    slowArray = concat(slowArray, floatArray);
                }
            }

            mat.release();
        }

        for(int i = 0; i < fastArray.length; i++) {
            fastArray[i] = (fastArray[i] - 0.45f) / 0.225f;
        }

        for(int i = 0; i < slowArray.length; i++) {
            slowArray[i] = (slowArray[i] - 0.45f) / 0.225f;
        }

        INDArray ndFastArray = Nd4j.create(fastArray, new int[] {32, 256, 341, 3}, 'c');
        INDArray pndFastArray = ndFastArray.permute(3, 0, 1, 2);

        INDArray ndSlowArray = Nd4j.create(slowArray, new int[] {8, 256, 341, 3}, 'c');
        INDArray pndSlowArray = ndSlowArray.permute(3, 0, 1, 2);

        INDArray flattenFastArray = Nd4j.toFlattened('c', pndFastArray);
        Tensor fastTensor = Tensor.fromBlob(flattenFastArray.toFloatVector(), new long[] {1, 3, 32, 256, 341});

        INDArray flattenSlowArray = Nd4j.toFlattened('c', pndSlowArray);
        Tensor slowTensor = Tensor.fromBlob(flattenSlowArray.toFloatVector(), new long[] {1, 3, 8, 256, 341});

        return new Tensor[] {slowTensor, fastTensor};
    }

    public static int SIZE = 256;
    private static Mat scale(Mat source) {
        int height = source.rows();
        int width = source.cols();

        if ((width <= height && width == SIZE) || (
             height <= width && height == SIZE)) {
            return source;
        }

        int newHeight = SIZE;
        int newWidth = SIZE;
        if (width < height)
            newHeight = (int)(Math.floor(((float)height) * SIZE / width));
        else
            newWidth = (int)(Math.floor(((float)width) * SIZE / height));

        Mat resized = new Mat();
        Imgproc.resize(source, resized, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LINEAR);

        return resized;
    }

    private static float[] concat(float[] first, float[] second) {
        float[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }
}
