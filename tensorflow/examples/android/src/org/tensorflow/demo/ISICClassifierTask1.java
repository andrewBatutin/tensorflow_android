package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import com.rewintous.isic.model.MaskRCNNModel;
import com.rewintous.isic.utils.dto.DetectionResult;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Collections;
import java.util.List;

public class ISICClassifierTask1 implements Classifier {
    private static final String TAG = "ISICClassifierTask1";
    private TensorFlowInferenceInterface inferenceInterface;



    public static Classifier create(
            AssetManager assetManager,
            String modelFilename) {
        ISICClassifierTask1 c = new ISICClassifierTask1();
        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        return c;
    }


    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        MaskRCNNModel model = new MaskRCNNModel();
        List<DetectionResult> detect = model.detect(inferenceInterface, bitmap);

/*
        int i = 0;
        for (DetectionResult detectionResult : detect) {
            i++;
            Bitmap bitmapOut = maskAsBWImage(detectionResult.getMask());
            System.out.println("bitmap = " + bitmap);
        }*/
        return Collections.emptyList();
    }

    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {

    }
}
