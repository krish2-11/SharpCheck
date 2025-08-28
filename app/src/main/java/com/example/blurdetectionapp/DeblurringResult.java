package com.example.blurdetectionapp;

import java.util.List;

class DeblurringResult {
    private android.graphics.Bitmap processedImage;
    private List<DeblurredObject> deblurredObjects;
    private boolean success;

    public DeblurringResult(android.graphics.Bitmap processedImage,
                            List<DeblurredObject> deblurredObjects, boolean success) {
        this.processedImage = processedImage;
        this.deblurredObjects = deblurredObjects;
        this.success = success;
    }

    public android.graphics.Bitmap getProcessedImage() { return processedImage; }
    public List<DeblurredObject> getDeblurredObjects() { return deblurredObjects; }
    public boolean isSuccess() { return success; }
    public int getDeblurredObjectCount() { return deblurredObjects.size(); }
}
