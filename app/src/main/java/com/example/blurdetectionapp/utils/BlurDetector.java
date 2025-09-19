package com.example.blurdetectionapp.utils;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

/**
 * Optimized Blur Detector for real-time document capture.
 * Works directly with OpenCV Mat (no Bitmap overhead).
 * Uses Laplacian variance + adaptive edge density checks.
 */
public class BlurDetector {
    private static final String TAG = "BlurDetector";

    private static final int PATCH_SIZE = 32; // smaller patch size for finer detection
    private static final double LAPLACIAN_THRESHOLD = 150.0; // tuned threshold for blur detection
    private static final double TEXTURE_THRESHOLD = 1.5; // low texture threshold for occlusion
    private static final double BLUR_PERCENTAGE_THRESHOLD = 0.5; // 50% patches blurred triggers blur flag
    private static final double OCCLUSION_PERCENTAGE_THRESHOLD = 0.08; // 8% patches occluded triggers occlusion flag

    public static class BlurDetectionResult {
        public final double avgVariance;
        public final double blurPercentage;
        public final double occlusionPercentage;
        public final boolean isBlurred;
        public final boolean isOccluded;

        public BlurDetectionResult(double avgVariance, double blurPct, double occPct, boolean blurred, boolean occluded) {
            this.avgVariance = avgVariance;
            this.blurPercentage = blurPct;
            this.occlusionPercentage = occPct;
            this.isBlurred = blurred;
            this.isOccluded = occluded;
        }
    }

    /**
     * Detects blur and occlusion in the given image Mat.
     * @param mat Input image in RGBA format.
     * @return BlurDetectionResult containing metrics and flags.
     */
    public static BlurDetectionResult detectBlurAndOcclusion(Mat mat) {
        if (mat == null || mat.empty()) {
            Log.w(TAG, "Input Mat is null or empty. Returning default blurred and occluded result.");
            return new BlurDetectionResult(0, 1.0, 1.0, true, true);
        }

        // Convert input image to grayscale
        Mat gray = new Mat();
        try {
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY);
        } catch (Exception e) {
            Log.e(TAG, "Failed to convert input Mat to grayscale: " + e.getMessage());
            gray.release();
            return new BlurDetectionResult(0, 1.0, 1.0, true, true);
        }

        MatOfDouble globalMean = new MatOfDouble();
        MatOfDouble globalStddev = new MatOfDouble();
        Core.meanStdDev(gray, globalMean, globalStddev);
        double globalTexture = globalStddev.get(0, 0)[0];
        globalMean.release();
        globalStddev.release();
        if (globalTexture < 5.0) {
            // Image is uniform color, treat as not blurred and not occluded
            gray.release();
            Log.d(TAG, "Image is uniform color (low global texture). Treating as not blurred.");
            return new BlurDetectionResult(0, 0.0, 0.0, false, false);
        }

        int rows = gray.rows();
        int cols = gray.cols();

        if (rows == 0 || cols == 0) {
            Log.w(TAG, "Grayscale Mat has zero rows or columns. Returning default result.");
            gray.release();
            return new BlurDetectionResult(0, 1.0, 1.0, true, true);
        }

        int blurPatches = 0;
        int occlusionPatches = 0;
        int totalPatches = 0;
        double totalVariance = 0.0;

        // Calculate global median intensity to adapt edge detection thresholds
        double medianIntensity = calculateMedianIntensity(gray);
        double cannyThreshold1 = Math.max(50, medianIntensity * 0.66);
        double cannyThreshold2 = Math.max(150, medianIntensity * 1.33);

        Log.d(TAG, String.format("Adaptive Canny thresholds: threshold1=%.1f, threshold2=%.1f, medianIntensity=%.1f",
                cannyThreshold1, cannyThreshold2, medianIntensity));

        for (int y = 0; y < rows; y += PATCH_SIZE) {
            for (int x = 0; x < cols; x += PATCH_SIZE) {
                int width = Math.min(PATCH_SIZE, cols - x);
                int height = Math.min(PATCH_SIZE, rows - y);
                Rect roi = new Rect(x, y, width, height);
                Mat patch = new Mat(gray, roi);

                if (patch.empty()) {
                    Log.w(TAG, String.format("Skipping empty patch at x=%d, y=%d", x, y));
                    patch.release();
                    continue;
                }

                try {
                    // Calculate Laplacian variance for blur detection
                    Mat lap = new Mat();
                    Imgproc.Laplacian(patch, lap, CvType.CV_64F);
                    MatOfDouble mean = new MatOfDouble();
                    MatOfDouble stddev = new MatOfDouble();
                    Core.meanStdDev(lap, mean, stddev);
                    double variance = stddev.get(0, 0)[0];
                    variance = variance * variance;
                    totalVariance += variance;

                    if (variance < LAPLACIAN_THRESHOLD) {
                        blurPatches++;
                    }

                    // Texture measure: standard deviation of patch intensity
                    MatOfDouble patchMean = new MatOfDouble();
                    MatOfDouble patchStddev = new MatOfDouble();
                    Core.meanStdDev(patch, patchMean, patchStddev);
                    double texture = patchStddev.get(0, 0)[0];

                    // Edge density calculation using adaptive Canny thresholds
                    Mat edges = new Mat();
                    Imgproc.Canny(patch, edges, cannyThreshold1, cannyThreshold2);
                    int edgePixels = Core.countNonZero(edges);
                    double edgeDensity = (double) edgePixels / (width * height);
                    edges.release();

                    // Consider patch occluded if texture is low OR edge density is low
                    if (texture < TEXTURE_THRESHOLD || edgeDensity < 0.1) {
                        occlusionPatches++;
                    }

                    totalPatches++;

                    lap.release();
                } catch (Exception e) {
                    Log.e(TAG, String.format("Exception processing patch at x=%d, y=%d: %s", x, y, e.getMessage()));
                } finally {
                    patch.release();
                }
            }
        }

        gray.release();

        if (totalPatches == 0) {
            Log.w(TAG, "No patches processed. Returning default blurred and occluded result.");
            return new BlurDetectionResult(0, 1.0, 1.0, true, true);
        }

        double avgVariance = totalVariance / totalPatches;
        double blurPercentage = (double) blurPatches / totalPatches;
        double occlusionPercentage = (double) occlusionPatches / totalPatches;

        boolean isBlurred = blurPercentage > BLUR_PERCENTAGE_THRESHOLD;
        boolean isOccluded = occlusionPercentage > OCCLUSION_PERCENTAGE_THRESHOLD;

        Log.d(TAG, String.format(
                "Blur detection result: AvgVariance=%.2f, BlurPct=%.2f%%, OcclusionPct=%.2f%%, Blurred=%b, Occluded=%b",
                avgVariance, blurPercentage * 100.0, occlusionPercentage * 100.0, isBlurred, isOccluded));

        return new BlurDetectionResult(avgVariance, blurPercentage, occlusionPercentage, isBlurred, isOccluded);
    }

    /**
     * Calculates the median intensity of a grayscale Mat.
     * @param grayMat Grayscale Mat.
     * @return Median pixel intensity.
     */
    private static double calculateMedianIntensity(Mat grayMat) {
        if (grayMat == null || grayMat.empty()) {
            return 0.0;
        }

        Mat sorted = new Mat();
        try {
            grayMat.reshape(0, 1).copyTo(sorted);
            Core.sort(sorted, sorted, Core.SORT_ASCENDING);
            int mid = sorted.cols() / 2;
            if (sorted.cols() % 2 == 0) {
                double median = (sorted.get(0, mid - 1)[0] + sorted.get(0, mid)[0]) / 2.0;
                return median;
            } else {
                return sorted.get(0, mid)[0];
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to calculate median intensity: " + e.getMessage());
            return 0.0;
        } finally {
            sorted.release();
        }
    }
}
