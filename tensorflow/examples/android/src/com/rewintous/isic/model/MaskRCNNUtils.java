package com.rewintous.isic.model;

import android.graphics.Bitmap;
import android.graphics.Color;
import com.rewintous.isic.utils.dto.FloatBox;
import com.rewintous.isic.utils.numpy.NumpyCustomOperations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class MaskRCNNUtils {
    private MaskRCNNUtils() {
    }

    /**
     * @param scales        1D array of anchor sizes in pixels. Example: [32, 64, 128]
     * @param ratios        1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
     * @param width         of the feature map over which to generate anchors.
     * @param height        of the feature map over which to generate anchors.
     * @param featureStride Stride of the feature map relative to the image in pixels.
     * @param anchorStride  Stride of anchors on the feature map. For example, if the
     */
    public static List<FloatBox> generateAnchors(float[] scales, float[] ratios, int width, int height, int featureStride, float anchorStride) {
        INDArray scalesArray = Nd4j.create(scales);
        INDArray ratiosArray = Nd4j.create(ratios);

        INDArray[] scalesRatios = Nd4j.meshgrid(scalesArray, ratiosArray);
        scalesArray = Nd4j.toFlattened(scalesRatios[0]);
        ratiosArray = Nd4j.toFlattened(scalesRatios[1]);

        // Enumerate heights and widths from scales and ratios
        INDArray heights = scalesArray.div(Transforms.sqrt(ratiosArray));
        INDArray widths = scalesArray.mul(Transforms.sqrt(ratiosArray));

        float[][] heightWidth = Nd4j.vstack(heights, widths).toFloatMatrix();

        // Enumerate shifts in feature space
        INDArray shiftsY = NumpyCustomOperations.arange(0, height, anchorStride).mul(featureStride);
        INDArray shiftsX = NumpyCustomOperations.arange(0, width, anchorStride).mul(featureStride);

        List<FloatBox> boxes = new LinkedList<>();

        for (float shiftX : shiftsX.toFloatVector()) {
            for (float shiftY : shiftsY.toFloatVector()) {
                for (int i = 0; i < heights.columns(); i++) {
                    float h = heightWidth[0][i];
                    float w = heightWidth[1][i];
                    boxes.add(new FloatBox((float) (shiftX - 0.5 * w),
                            (float) (shiftY - 0.5 * h),
                            (float) (shiftX + 0.5 * w),
                            (float) (shiftY + 0.5 * h)));
                }
            }
        }

        return boxes;
    }

    /**
     * @param scales         1D array of anchor sizes in pixels. Example: [32, 64, 128]
     * @param ratios         1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
     * @param widths         of the feature map over which to generate anchors.
     * @param heights        of the feature map over which to generate anchors.
     * @param featureStrides of the feature map relative to the image in pixels.
     * @param anchorStride   Stride of anchors on the feature map. For example, if the
     */
    public static List<FloatBox> generatePyramidAnchors(int[] scales, float[] ratios, int[] widths, int[] heights, int[] featureStrides, float anchorStride) {
        List<FloatBox> allAnchors = new LinkedList<>();
        for (int i = 0; i < scales.length; i++) {
            int scale = scales[i];
            allAnchors.addAll(generateAnchors(new float[]{scale}, ratios, widths[i], heights[i], featureStrides[i], anchorStride));
        }
        return allAnchors;
    }

    /**
     * """Takes attributes of an image and puts them in one 1D array.
     * <p>
     * image_id:
     * image_shape:
     * window:
     * <p>
     * scale:
     * active_class_ids:
     *
     * @param image_id             An int ID of the image. Useful for debugging.
     *                             *             original_image_shape: [H, W, C] before resizing or padding.
     * @param original_image_shape [H, W, C] after resizing and padding
     * @param image_shape          (y1, x1, y2, x2) in pixels. The area of the image where the real
     * @param window               image is (excluding the padding)
     * @param scale                The scaling factor applied to the original image (float32)
     * @param active_class_ids     List of class_ids available in the dataset from which
     *                             *     the image came. Useful if training on images from multiple datasets
     *                             *     where not all classes are present in all datasets.
     * @return
     */
    public static float[] compose_image_meta(int image_id,
                                             int[] original_image_shape,
                                             int[] image_shape,
                                             FloatBox window,
                                             float scale,
                                             int[] active_class_ids) {
        /*
        meta = np.array(
            [image_id] +                  # size=1
    list(original_image_shape) +  # size=3
    list(image_shape) +           # size=3
    list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
    list(active_class_ids)        # size=num_classes
    )
         */
        //should be 12 + NUM_CLASSES components including background
        List<Float> singleList = new ArrayList<>();
        singleList.add((float) image_id);

        singleList.add((float) original_image_shape[0]);
        singleList.add((float) original_image_shape[1]);
        singleList.add((float) original_image_shape[2]);

        singleList.add((float) image_shape[0]);
        singleList.add((float) image_shape[1]);
        singleList.add((float) image_shape[2]);

        singleList.add(window.getY1());
        singleList.add(window.getX1());
        singleList.add(window.getY2());
        singleList.add(window.getX2());

        singleList.add(scale);

        for (int id : active_class_ids) {
            singleList.add((float) id);
        }

        float[] output = new float[singleList.size()];
        Iterator<Float> it = singleList.iterator();
        for (int i = 0; i < output.length; i++) {
            output[i] = it.next();
        }

        return output;
    }

    /**
     * @param boxes  boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
     * @param width
     * @param height
     * @return [N, (y1, x1, y2, x2)] in normalized coordinates
     */
    public static List<FloatBox> normBoxes(List<FloatBox> boxes, int width, int height) {
        List<FloatBox> result = new ArrayList<>();
        for (FloatBox box : boxes) {
            FloatBox r = box.copy();
            r.normalize(width, height);
            result.add(r);
        }
        return result;
    }

    /**
     * @param boxes  boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
     * @param width
     * @param height
     * @return [N, (y1, x1, y2, x2)] in int coordinates
     */
    public static List<FloatBox> denormBoxes(List<FloatBox> boxes, int width, int height) {
        List<FloatBox> result = new ArrayList<>();
        for (FloatBox box : boxes) {
            FloatBox r = box.copy();
            r.denormalize(width, height);
            result.add(r);
        }
        return result;
    }

    public static void main(String[] args) {
        generateAnchors(new float[]{32}, new float[]{0.5f, 1, 2}, 192, 192, 4, 1);
    }

    /**
     * @param mask [H,W]
     * @return serializable buffered image
     */
    public static Bitmap maskAsBWImage(boolean[][] mask) {
        int width = mask[0].length;
        int height = mask.length;
        Bitmap bufferedImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (mask[h][w]) {
                    bufferedImage.setPixel(w,h, Color.WHITE);
                }
            }
        }

        return bufferedImage;
    }
}
