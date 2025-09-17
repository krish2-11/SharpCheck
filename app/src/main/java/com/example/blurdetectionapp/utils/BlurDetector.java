package com.example.blurdetectionapp.utils;

import static org.opencv.android.NativeCameraView.TAG;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

/**
 * Optimized Blur Detector for real-time document capture.
 * Works directly with OpenCV Mat (no Bitmap overhead).
 * Uses Laplacian variance + edge density checks.
 */
public class BlurDetector {
    private static final int PATCH_SIZE = 32; // smaller patch size for finer detection
    private static final double LAPLACIAN_THRESHOLD = 200.0; // tuned threshold
    private static final double TEXTURE_THRESHOLD = 1.5; // low texture threshold for occlusion
    private static final double BLUR_PERCENTAGE_THRESHOLD = 0.5;
    private static final double OCCLUSION_PERCENTAGE_THRESHOLD = 0.08; // 5% patches occluded

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

    public static BlurDetectionResult detectBlurAndOcclusion(Mat mat) {
        if (mat == null || mat.empty()) {
            return new BlurDetectionResult(0, 100, 100, true, true);
        }

        Mat gray = new Mat();
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY);

        int rows = gray.rows();
        int cols = gray.cols();
        int blurPatches = 0;
        int occlusionPatches = 0;
        int totalPatches = 0;
        double totalVariance = 0.0;

        for (int y = 0; y < rows; y += PATCH_SIZE) {
            for (int x = 0; x < cols; x += PATCH_SIZE) {
                int width = Math.min(PATCH_SIZE, cols - x);
                int height = Math.min(PATCH_SIZE, rows - y);
                Rect roi = new Rect(x, y, width, height);
                Mat patch = new Mat(gray, roi);

                // Laplacian variance
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

                // Edge density calculation using Canny
                Mat edges = new Mat();
                Imgproc.Canny(patch, edges, 50, 150);
                int edgePixels = Core.countNonZero(edges);
                double edgeDensity = (double) edgePixels / (width * height);
                edges.release();

                // Consider patch occluded if texture is low OR edge density is low
                if (texture < TEXTURE_THRESHOLD || edgeDensity < 0.1) {  // 0.1 is a tunable threshold
                    occlusionPatches++;
                }

                totalPatches++;

                lap.release();
                patch.release();
            }
        }

        double avgVariance = totalVariance / totalPatches;
        double blurPercentage = (double) blurPatches / totalPatches;
        double occlusionPercentage = (double) occlusionPatches / totalPatches;

        boolean isBlurred = blurPercentage > BLUR_PERCENTAGE_THRESHOLD;
        boolean isOccluded = occlusionPercentage > OCCLUSION_PERCENTAGE_THRESHOLD;

        Log.d(TAG, String.format("Blur: AvgVariance=%.1f, BlurPct=%.1f%%, OcclusionPct=%.1f%%, Blurred=%b, Occluded=%b",
                avgVariance, blurPercentage * 100.0, occlusionPercentage * 100.0, isBlurred, isOccluded));

        gray.release();
        return new BlurDetectionResult(avgVariance, blurPercentage, occlusionPercentage, isBlurred, isOccluded);
    }

}
