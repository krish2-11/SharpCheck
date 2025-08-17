package com.example.blurdetectionapp;

public class BlurResult {

    private final boolean blurred;
    private final double laplacianVariance;
    private final double tenengradScore;
    private final double edgeDensity;

    public BlurResult(boolean blurred, double laplacianVariance, double tenengradScore, double edgeDensity) {
        this.blurred = blurred;
        this.laplacianVariance = laplacianVariance;
        this.tenengradScore = tenengradScore;
        this.edgeDensity = edgeDensity;
    }

    public boolean isBlurred() {
        return blurred;
    }

    public double getLaplacianVariance() {
        return laplacianVariance;
    }

    public double getTenengradScore() {
        return tenengradScore;
    }

    public double getEdgeDensity() {
        return edgeDensity;
    }

    @Override
    public String toString() {
        return String.format("BlurResult{blurred=%s, laplacian=%.2f, tenengrad=%.2f, edgeDensity=%.4f}",
                blurred, laplacianVariance, tenengradScore, edgeDensity);
    }
}