package com.rewintous.isic.utils.dto;

public class DetectionResult {
    private final FloatBox boundingBox;
    private final int classId;
    private final float score;
    //h,w
    private final boolean [][] mask;

    public DetectionResult(FloatBox boundingBox, int classId, float score, boolean[][] mask) {
        this.boundingBox = boundingBox;
        this.classId = classId;
        this.score = score;
        this.mask = mask;
    }

    public FloatBox getBoundingBox() {
        return boundingBox;
    }

    public int getClassId() {
        return classId;
    }

    public float getScore() {
        return score;
    }

    public boolean[][] getMask() {
        return mask;
    }

}
