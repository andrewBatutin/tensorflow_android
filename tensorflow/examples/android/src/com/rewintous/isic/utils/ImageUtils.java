package com.rewintous.isic.utils;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import com.rewintous.isic.utils.consts.ResizeMode;
import com.rewintous.isic.utils.dto.Padding;
import com.rewintous.isic.utils.dto.Rectangle;
import com.rewintous.isic.utils.dto.ResizeImageResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.Math.*;

public class ImageUtils {
    private ImageUtils() {
    }

    /**
     * Resizes an image keeping the aspect ratio unchanged.
     *
     * @param image
     * @param minDim     if not null, resizes the image such that it's smaller dimension == min_dim
     * @param maxDim     if provided, ensures that the image longest side doesn't exceed this value.
     * @param minScale   if provided, ensure that the image is scaled up by at least this percent even if min_dim doesn't require it.
     * @param resizeMode none: No resizing. Return the image unchanged.
     *                   square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
     *                   pad64: Pads width and height with zeros to make them multiples of 64.
     *                   If min_dim or min_scale are provided, it scales the image up before padding. max_dim is ignored in this mode.
     *                   The multiple of 64 is needed to ensure smooth scaling of feature maps up and down the 6 levels of the FPN pyramid (2**6=64)
     *                   crop: Picks random crops from the image. First, scales the image based
     *                   on min_dim and min_scale, then picks a random crop of
     *                   size min_dim x min_dim. Can be used in training only.
     *                   max_dim is not used in this mode.
     * @return image: the resized image
     * window: (y1, x1, y2, x2). If max_dim is provided, padding might
     * be inserted in the returned image. If so, this window is the
     * coordinates of the image part of the full image (excluding
     * the padding). The x2, y2 pixels are not included.
     * scale: The scale factor used to resize the image
     * padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
     */
    public static ResizeImageResult resize_image(Bitmap image, Integer minDim, Integer maxDim, Float minScale, ResizeMode resizeMode) {
//        image_dtype = image.dtype
        // Default window (y1, x1, y2, x2) and default scale == 1.
        int h = image.getHeight();
        int w = image.getWidth();
        Rectangle window = new Rectangle(0, 0, w, h);

        float scale = 1;
        Padding padding = new Padding();

        if (resizeMode == ResizeMode.NONE) {
            return new ResizeImageResult(image, window, scale, padding);
        }

        if (minDim != null) {
            scale = max(1, (float) minDim / min(h, w));
        }

        if (minScale != null && scale < minScale) {
            scale = minScale;
        }

        if (maxDim != null && resizeMode == ResizeMode.SQUARE) {
            int imageMax = max(h, w);
            if (round(imageMax * scale) > maxDim) {
                scale = (float) maxDim / imageMax;
            }
        }

        if (scale != 1) {
            image = resizeImageWithHint(image, round(w * scale), round(h * scale), 0);
        }

        h = image.getHeight();
        w = image.getWidth();
        switch (resizeMode) {
            case SQUARE:
                padding.setTop((maxDim - h) / 2);
                padding.setBottom(maxDim - h - padding.getTop());
                padding.setLeft((maxDim - w) / 2);
                padding.setRight(maxDim - w - padding.getLeft());
                window.setBounds(padding.getLeft(), padding.getTop(), w, h);
                image = padBufferedImage(image, padding);
                break;
            case PAD64:
                if (minDim % 64 != 0) {
                    throw new AssertionError("Minimum dimension must be a multiple of 64");
                }

                //fixme ugly duplicate below
                int topPad, bottomPad;
                if (h % 64 > 0) {
                    int maxH = h - (h % 64) + 64;
                    topPad = (maxH - h) / 2;
                    bottomPad = maxH - h - topPad;
                } else {
                    topPad = bottomPad = 0;
                }

                int leftPad, rightPad;

                if (w % 64 > 0) {
                    int maxW = w - (w % 64) + 64;
                    leftPad = (maxW - w) / 2;
                    rightPad = maxW - w - leftPad;
                } else {
                    leftPad = rightPad = 0;
                }

                padding.setLeft(leftPad);
                padding.setRight(rightPad);
                padding.setTop(topPad);
                padding.setBottom(bottomPad);
                image = padBufferedImage(image, padding);
                window.setBounds(padding.getLeft(), padding.getTop(), w, h);
                break;
            default:
                throw new IllegalArgumentException("Unknown resize mode: " + resizeMode);

        }

        return new ResizeImageResult(image, window, scale, padding);
    }

    public static Bitmap resizeImageWithHint(Bitmap originalImage, int newWidth, int newHeight, int type) {
        return Bitmap.createScaledBitmap(originalImage, newWidth, newHeight, true);
    }

    public static Bitmap padBufferedImage(Bitmap originalImage, Padding padding) {
        Bitmap outputImage = Bitmap.createBitmap(originalImage.getWidth() + padding.getLeft() + padding.getRight(),
                originalImage.getHeight() + padding.getTop() + padding.getBottom(), Bitmap.Config.ARGB_8888);

        Canvas canvas = new Canvas(outputImage);
        canvas.drawBitmap(originalImage, padding.getLeft(), padding.getTop(), null);

        return outputImage;
    }

    public static INDArray asMatrix(Bitmap bufferedImage) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        INDArray indArray = Nd4j.zeros(height, width, 3);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int rgb = bufferedImage.getPixel(w, h);
                indArray.putScalar(h, w, 0, Color.red(rgb));
                indArray.putScalar(h, w, 1, Color.green(rgb));
                indArray.putScalar(h, w, 2, Color.blue(rgb));
//                indArray.putScalar(h, w, 0, (rgb >> 16) & 0xFF);
//                indArray.putScalar(h, w, 1, (rgb >> 8)& 0xFF);
//                indArray.putScalar(h, w, 2, rgb & 0xFF);
            }
        }
        return indArray;
    }
}
