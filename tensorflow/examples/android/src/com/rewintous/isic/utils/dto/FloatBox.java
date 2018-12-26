package com.rewintous.isic.utils.dto;

public class FloatBox {
    private float x1;
    private float y1;
    private float x2;
    private float y2;

    public FloatBox(Rectangle r) {
        this((float)r.getX(),
                (float)r.getY(),
                (float)(r.getX() +  r.getWidth()),
                (float) (r.getY() + r.getHeight()));
    }

    public FloatBox(float x1, float y1, float x2, float y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    public float getX1() {
        return x1;
    }

    public float getY1() {
        return y1;
    }

    public float getX2() {
        return x2;
    }

    public float getY2() {
        return y2;
    }

    public float getArea() {
        return getHeight() * getWidth();
    }

    public float getWidth() {
        return x2 - x1;
    }

    public float getHeight() {
        return y2 - y1;
    }

    public void normalize(int width, int height) {
        this.x1 /= (float) (width - 1);
        this.y1 /= (float) (height - 1);
        this.x2 -= 1;
        this.x2 /= (float) (width - 1);
        this.y2 -= 1;
        this.y2 /= (float) (height - 1);
    }

    public void denormalize(int width, int height) {
        this.x1 *= (float) (width - 1);
        this.y1 *= (float) (height - 1);
        this.x2 *= (float) (width - 1);
        this.y2 *= (float) (height - 1);
        this.x2 += 1;
        this.y2 += 1;
    }

    public FloatBox copy() {
        return new FloatBox(this.x1, this.y1, this.x2, this.y2);
    }
}
