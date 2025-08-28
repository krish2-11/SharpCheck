package com.example.blurdetectionapp;

import org.opencv.core.Mat;

class DeblurredObject {
    private BlurredObject originalObject;
    private Mat deblurredRoi;
    private double qualityImprovement;

    public DeblurredObject(BlurredObject originalObject, Mat deblurredRoi, double qualityImprovement) {
        this.originalObject = originalObject;
        this.deblurredRoi = deblurredRoi;
        this.qualityImprovement = qualityImprovement;
    }

    public BlurredObject getOriginalObject() { return originalObject; }
    public Mat getDeblurredRoi() { return deblurredRoi; }
    public double getQualityImprovement() { return qualityImprovement; }
}
