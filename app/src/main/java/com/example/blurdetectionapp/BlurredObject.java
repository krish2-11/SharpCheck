package com.example.blurdetectionapp;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

class BlurredObject {
    private Rect boundingRect;
    private Mat grayRoi;
    private Mat colorRoi;
    private double area;
    private double perimeter;
    private double circularity;
    private double aspectRatio;
    private String detectionMethod;

    // Blur metrics
    private double laplacianVariance;
    private double sobelMagnitude;
    private double pixelVariance;
    private double tenengrad;
    private boolean isBlurred;

    public BlurredObject(Rect boundingRect, Mat grayRoi, Mat colorRoi,
                         double area, double perimeter, double circularity,
                         double aspectRatio, String detectionMethod) {
        this.boundingRect = boundingRect;
        this.grayRoi = grayRoi;
        this.colorRoi = colorRoi;
        this.area = area;
        this.perimeter = perimeter;
        this.circularity = circularity;
        this.aspectRatio = aspectRatio;
        this.detectionMethod = detectionMethod;
        this.isBlurred = false;
    }

    // Getters
    public Rect getBoundingRect() { return boundingRect; }
    public Mat getGrayRoi() { return grayRoi; }
    public Mat getColorRoi() { return colorRoi; }
    public double getArea() { return area; }
    public double getPerimeter() { return perimeter; }
    public double getCircularity() { return circularity; }
    public double getAspectRatio() { return aspectRatio; }
    public String getDetectionMethod() { return detectionMethod; }
    public double getLaplacianVariance() { return laplacianVariance; }
    public double getSobelMagnitude() { return sobelMagnitude; }
    public double getPixelVariance() { return pixelVariance; }
    public double getTenengrad() { return tenengrad; }
    public boolean isBlurred() { return isBlurred; }

    // Setters
    public void setBlurMetrics(double laplacianVariance, double sobelMagnitude,
                               double pixelVariance, double tenengrad) {
        this.laplacianVariance = laplacianVariance;
        this.sobelMagnitude = sobelMagnitude;
        this.pixelVariance = pixelVariance;
        this.tenengrad = tenengrad;
    }

    public void setBlurred(boolean blurred) { this.isBlurred = blurred; }

    @Override
    public String toString() {
        return String.format("BlurredObject[method=%s, area=%.0f, blurred=%b, laplacian=%.2f]",
                detectionMethod, area, isBlurred, laplacianVariance);
    }
}
