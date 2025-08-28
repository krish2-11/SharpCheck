package com.example.blurdetectionapp;

import java.util.List;

class BlurredObjectDetectionResult {
    private List<BlurredObject> blurredObjects;
    private int totalObjectsDetected;

    public BlurredObjectDetectionResult(List<BlurredObject> blurredObjects, int totalObjectsDetected) {
        this.blurredObjects = blurredObjects;
        this.totalObjectsDetected = totalObjectsDetected;
    }

    public List<BlurredObject> getBlurredObjects() { return blurredObjects; }
    public int getTotalObjectsDetected() { return totalObjectsDetected; }
    public int getBlurredObjectCount() { return blurredObjects.size(); }
    public boolean hasBlurredObjects() { return !blurredObjects.isEmpty(); }
}
