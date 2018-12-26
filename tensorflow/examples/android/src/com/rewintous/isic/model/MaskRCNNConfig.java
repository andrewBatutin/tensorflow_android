package com.rewintous.isic.model;


import com.rewintous.isic.utils.consts.ResizeMode;

public class MaskRCNNConfig {
    private int numClasses = 1 + 1; //1 FG
    private int[] rpnAnchorScales = {32, 64, 128, 256, 512};
    private float[] rpnAnchorRatios = {0.5f, 1f, 2f};
    private int[] backboneStrides = {4, 8, 16, 32, 64};
    private int rpnAnchorStride = 1;
    private ResizeMode imageResizeMode = ResizeMode.SQUARE;
    private int imageMinDim = 768;
    private int imageMaxDim = 768;
    private float imageMinScale = 0;
//    private float [] meanPixel = Nd4j.create(new float[][]{{123.7f}, {116.8f}, {103.9f}});
    private float [] meanPixel = new float[]{123.7f, 116.8f, 103.9f};

    public int[] getBackboneStrides() {
        return backboneStrides;
    }

    //replace with Builder
    public void setBackboneStrides(int[] backboneStrides) {
        this.backboneStrides = backboneStrides;
    }

    public ResizeMode getImageResizeMode() {
        return imageResizeMode;
    }

    public void setImageResizeMode(ResizeMode imageResizeMode) {
        this.imageResizeMode = imageResizeMode;
    }

    public int getImageMinDim() {
        return imageMinDim;
    }

    public void setImageMinDim(int imageMinDim) {
        this.imageMinDim = imageMinDim;
    }

    public int getImageMaxDim() {
        return imageMaxDim;
    }

    public void setImageMaxDim(int imageMaxDim) {
        this.imageMaxDim = imageMaxDim;
    }

    public float getImageMinScale() {
        return imageMinScale;
    }

    public void setImageMinScale(float imageMinScale) {
        this.imageMinScale = imageMinScale;
    }

    public float[] getMeanPixel() {
        return meanPixel;
    }

    public void setMeanPixel(float[] meanPixel) {
        this.meanPixel = meanPixel;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public int[] getRpnAnchorScales() {
        return rpnAnchorScales;
    }

    public float[] getRpnAnchorRatios() {
        return rpnAnchorRatios;
    }

    public int getRpnAnchorStride() {
        return rpnAnchorStride;
    }
}
