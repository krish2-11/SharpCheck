package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

public class BlurDetector {

    private static final String TAG = "BlurDetector";

    // Optimized thresholds for better accuracy
    private static final double BLUR_THRESHOLD_LAPLACIAN = 50.0;  // Higher for stricter detection
    private static final double BLUR_THRESHOLD_TENENGRAD = 15.0;  // Higher for stricter detection
    private static final double BLUR_THRESHOLD_EDGE_DENSITY = 0.05; // New metric for edge density

    // Weights for combined blur score
    private static final double WEIGHT_LAPLACIAN = 0.4;
    private static final double WEIGHT_TENENGRAD = 0.3;
    private static final double WEIGHT_EDGE_DENSITY = 0.3;

    /**
     * Main blur detection method using multiple algorithms
     */
    public BlurResult detectBlur(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            Log.e(TAG, "Invalid bitmap for blur detection");
            return new BlurResult(true, 0, 0, 0); // Consider invalid images as blurred
        }

        Mat rgba = null;
        Mat gray = null;

        try {
            Log.d(TAG, "Starting blur detection on image: " + bitmap.getWidth() + "x" + bitmap.getHeight());

            // Convert bitmap to OpenCV Mat
            rgba = new Mat();
            Utils.bitmapToMat(bitmap, rgba);

            gray = new Mat();
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);

            // Calculate multiple blur metrics
            double laplacianVar = calculateLaplacianVariance(gray);
            double tenengradScore = calculateTenengradScore(gray);
            double edgeDensity = calculateEdgeDensity(gray);

            // Advanced blur detection using weighted combination
            boolean isBlurred = determineBlur(laplacianVar, tenengradScore, edgeDensity);

            BlurResult result = new BlurResult(isBlurred, laplacianVar, tenengradScore, edgeDensity);

            Log.d(TAG, String.format("Blur metrics - Laplacian: %.2f, Tenengrad: %.2f, Edge Density: %.4f, Result: %s",
                    laplacianVar, tenengradScore, edgeDensity, isBlurred ? "BLURRED" : "SHARP"));

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error in blur detection", e);
            return new BlurResult(false, 0, 0, 0); // Default to not blurred if detection fails
        } finally {
            if (rgba != null) rgba.release();
            if (gray != null) gray.release();
        }
    }

    /**
     * Calculate Laplacian variance - measures focus quality
     */
    private double calculateLaplacianVariance(Mat gray) {
        Mat laplacian = null;
        try {
            laplacian = new Mat();
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

            MatOfDouble mean = new MatOfDouble();
            MatOfDouble stddev = new MatOfDouble();
            Core.meanStdDev(laplacian, mean, stddev);

            double variance = stddev.get(0, 0)[0];
            return variance * variance;

        } catch (Exception e) {
            Log.e(TAG, "Error calculating Laplacian variance", e);
            return BLUR_THRESHOLD_LAPLACIAN + 1; // Return value above threshold
        } finally {
            if (laplacian != null) laplacian.release();
        }
    }

    /**
     * Calculate Tenengrad score - measures gradient magnitude
     */
    private double calculateTenengradScore(Mat gray) {
        Mat gradX = null;
        Mat gradY = null;
        Mat gradMag = null;
        try {
            gradX = new Mat();
            gradY = new Mat();

            // Use Sobel operator to compute gradients
            Imgproc.Sobel(gray, gradX, CvType.CV_64F, 1, 0, 3);
            Imgproc.Sobel(gray, gradY, CvType.CV_64F, 0, 1, 3);

            // Calculate gradient magnitude
            gradMag = new Mat();
            Core.magnitude(gradX, gradY, gradMag);

            // Calculate mean gradient magnitude
            return Core.mean(gradMag).val[0];

        } catch (Exception e) {
            Log.e(TAG, "Error calculating Tenengrad score", e);
            return BLUR_THRESHOLD_TENENGRAD + 1; // Return value above threshold
        } finally {
            if (gradX != null) gradX.release();
            if (gradY != null) gradY.release();
            if (gradMag != null) gradMag.release();
        }
    }

    /**
     * Calculate edge density - measures proportion of edge pixels
     */
    private double calculateEdgeDensity(Mat gray) {
        Mat edges = null;
        try {
            edges = new Mat();

            // Apply Canny edge detection
            Imgproc.Canny(gray, edges, 50, 150);

            // Count edge pixels
            int totalPixels = (int)(gray.rows() * gray.cols());
            int edgePixels = Core.countNonZero(edges);

            return (double)edgePixels / totalPixels;

        } catch (Exception e) {
            Log.e(TAG, "Error calculating edge density", e);
            return BLUR_THRESHOLD_EDGE_DENSITY + 0.01; // Return value above threshold
        } finally {
            if (edges != null) edges.release();
        }
    }

    /**
     * Advanced blur determination using weighted combination of metrics
     */
    private boolean determineBlur(double laplacianVar, double tenengradScore, double edgeDensity) {
        // Individual threshold checks
        boolean laplacianBlur = laplacianVar < BLUR_THRESHOLD_LAPLACIAN;
        boolean tenengradBlur = tenengradScore < BLUR_THRESHOLD_TENENGRAD;
        boolean edgeDensityBlur = edgeDensity < BLUR_THRESHOLD_EDGE_DENSITY;

        // Calculate weighted blur score (0 = sharp, 1 = blurred)
        double blurScore = 0;

        if (laplacianBlur) blurScore += WEIGHT_LAPLACIAN;
        if (tenengradBlur) blurScore += WEIGHT_TENENGRAD;
        if (edgeDensityBlur) blurScore += WEIGHT_EDGE_DENSITY;

        // Consider image blurred if weighted score > 0.5
        boolean isBlurred = blurScore > 0.5;

        Log.d(TAG, String.format("Individual checks - Laplacian: %s, Tenengrad: %s, EdgeDensity: %s, Score: %.2f",
                laplacianBlur, tenengradBlur, edgeDensityBlur, blurScore));

        return isBlurred;
    }

    /**
     * Get blur detection thresholds for tuning
     */
    public String getThresholdInfo() {
        return String.format("Thresholds - Laplacian: %.1f, Tenengrad: %.1f, EdgeDensity: %.3f",
                BLUR_THRESHOLD_LAPLACIAN, BLUR_THRESHOLD_TENENGRAD, BLUR_THRESHOLD_EDGE_DENSITY);
    }
}