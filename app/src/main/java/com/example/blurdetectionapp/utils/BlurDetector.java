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
 * Uses Laplacian variance + edge density checks.
 */
public class BlurDetector {

    private static final String TAG = "BlurDetector";

    // Thresholds
    private static final double STRICT_BLUR_THRESHOLD = 600.0;   // catches subtle blur
    private static final double EDGE_DENSITY_THRESHOLD = 0.02;   // % of edges required in frame

    // Resize target for speed (keep aspect ratio)
    private static final int RESIZE_WIDTH = 320;

    public static class BlurDetectionResult {
        public final double laplacianVariance;
        public final double edgeDensity;
        public final boolean isBlurred;
        public final String description;
        public final double confidenceScore;

        public BlurDetectionResult(double variance, double edgeDensity,
                                   boolean blurred, String desc, double confidence) {
            this.laplacianVariance = variance;
            this.edgeDensity = edgeDensity;
            this.isBlurred = blurred;
            this.description = desc;
            this.confidenceScore = confidence;
        }
    }

    /**
     * Main blur detection method.
     * @param mat Input image in Mat (preferably YUV->Mat converted before calling)
     */
    public static BlurDetectionResult detectBlur(Mat mat) {
        if (mat == null || mat.empty()) {
            return new BlurDetectionResult(0, 0, true, "No image", 0);
        }

        Mat resized = new Mat();
        Mat gray = new Mat();
        Mat laplacian = new Mat();
        Mat edges = new Mat();

        try {
            // ✅ Resize for efficiency
            double aspect = (double) mat.height() / mat.width();
            int newHeight = (int) (RESIZE_WIDTH * aspect);
            Imgproc.resize(mat, resized, new org.opencv.core.Size(RESIZE_WIDTH, newHeight));

            // ✅ Convert to grayscale
            Imgproc.cvtColor(resized, gray, Imgproc.COLOR_RGBA2GRAY);

            // ✅ Laplacian variance (sharpness measure)
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
            MatOfDouble mean = new MatOfDouble();
            MatOfDouble stddev = new MatOfDouble();
            Core.meanStdDev(laplacian, mean, stddev);
            double variance = stddev.get(0, 0)[0];
            variance = variance * variance;

            // ✅ Edge density check (for occlusion cases like fingers)
            Imgproc.Canny(gray, edges, 50, 150);
            double edgePixels = Core.countNonZero(edges);
            double totalPixels = gray.rows() * gray.cols();
            double edgeDensity = edgePixels / totalPixels;

            // ✅ Multi-criteria decision
            boolean isBlurred = variance < STRICT_BLUR_THRESHOLD || edgeDensity < EDGE_DENSITY_THRESHOLD;

            // ✅ Confidence (scaled by both variance + edge density)
            double sharpnessScore = Math.min(1.0, variance / (STRICT_BLUR_THRESHOLD * 2));
            double edgeScore = Math.min(1.0, edgeDensity / (EDGE_DENSITY_THRESHOLD * 2));
            double confidence = (sharpnessScore * 0.6 + edgeScore * 0.4) * 100.0;

            String desc = getDescription(variance, edgeDensity, isBlurred);

            Log.d(TAG, String.format("Blur Analysis - Variance: %.1f, EdgeDensity: %.3f, Blurred: %b, Confidence: %.1f%%, Desc: %s",
                    variance, edgeDensity, isBlurred, confidence, desc));

            return new BlurDetectionResult(variance, edgeDensity, isBlurred, desc, confidence);

        } finally {
            resized.release();
            gray.release();
            laplacian.release();
            edges.release();
        }
    }

    private static String getDescription(double variance, double edgeDensity, boolean blurred) {
        if (blurred) {
            if (edgeDensity < EDGE_DENSITY_THRESHOLD * 0.5) {
                return "Too few details (possible obstruction)";
            } else if (variance < STRICT_BLUR_THRESHOLD * 0.3) {
                return "Severely blurred";
            } else {
                return "Blurred / low detail";
            }
        } else {
            if (variance > STRICT_BLUR_THRESHOLD * 2) {
                return "Very sharp";
            } else {
                return "Acceptably sharp";
            }
        }
    }
}
