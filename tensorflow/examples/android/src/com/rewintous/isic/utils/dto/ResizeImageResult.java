package com.rewintous.isic.utils.dto;

import android.graphics.Bitmap;

public class ResizeImageResult {
    private Bitmap resizedImage;
    private Rectangle window;
    private float scale;
    private Padding padding;

    public ResizeImageResult(Bitmap resizedImage, Rectangle window, float scale, Padding padding) {
        this.resizedImage = resizedImage;
        this.window = window;
        this.scale = scale;
        this.padding = padding;
    }

    public Bitmap getResizedImage() {
        return resizedImage;
    }

    public Rectangle getWindow() {
        return window;
    }

    public float getScale() {
        return scale;
    }

    public Padding getPadding() {
        return padding;
    }
}
