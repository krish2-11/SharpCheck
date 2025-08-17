package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import java.util.List;

public class ObjectDetectionResult {

    private final Bitmap originalImage;
    private final Bitmap processedImage;
    private final List<DetectedObject> detectedObjects;

    public ObjectDetectionResult(Bitmap originalImage, Bitmap processedImage, List<DetectedObject> detectedObjects) {
        this.originalImage = originalImage;
        this.processedImage = processedImage;
        this.detectedObjects = detectedObjects;
    }

    public Bitmap getOriginalImage() {
        return originalImage;
    }

    public Bitmap getProcessedImage() {
        return processedImage;
    }

    public List<DetectedObject> getDetectedObjects() {
        return detectedObjects;
    }

    public int getObjectCount() {
        return detectedObjects != null ? detectedObjects.size() : 0;
    }

    public boolean hasObjects() {
        return getObjectCount() > 0;
    }

    @Override
    public String toString() {
        return String.format("ObjectDetectionResult{objectCount=%d, hasProcessedImage=%s}",
                getObjectCount(), processedImage != null);
    }
}