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

//    private static final String TAG = "BlurDetector";
//
//    // Thresholds
//    private static final double STRICT_BLUR_THRESHOLD = 600.0;   // catches subtle blur
//    private static final double EDGE_DENSITY_THRESHOLD = 0.02;   // % of edges required in frame
//
//    // Resize target for speed (keep aspect ratio)
//    private static final int RESIZE_WIDTH = 320;
//
//    public static class BlurDetectionResult {
//        public final double laplacianVariance;
//        public final double edgeDensity;
//        public final boolean isBlurred;
//        public final String description;
//        public final double confidenceScore;
//
//        public BlurDetectionResult(double variance, double edgeDensity,
//                                   boolean blurred, String desc, double confidence) {
//            this.laplacianVariance = variance;
//            this.edgeDensity = edgeDensity;
//            this.isBlurred = blurred;
//            this.description = desc;
//            this.confidenceScore = confidence;
//        }
//    }
//
//    /**
//     * Main blur detection method.
//     * @param mat Input image in Mat (preferably YUV->Mat converted before calling)
//     */
//    public static BlurDetectionResult detectBlur(Mat mat) {
//        if (mat == null || mat.empty()) {
//            return new BlurDetectionResult(0, 0, true, "No image", 0);
//        }
//
//        Mat resized = new Mat();
//        Mat gray = new Mat();
//        Mat laplacian = new Mat();
//        Mat edges = new Mat();
//
//        try {
//            // ✅ Resize for efficiency
//            double aspect = (double) mat.height() / mat.width();
//            int newHeight = (int) (RESIZE_WIDTH * aspect);
//            Imgproc.resize(mat, resized, new org.opencv.core.Size(RESIZE_WIDTH, newHeight));
//
//            // ✅ Convert to grayscale
//            Imgproc.cvtColor(resized, gray, Imgproc.COLOR_RGBA2GRAY);
//
//            // ✅ Laplacian variance (sharpness measure)
//            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
//            MatOfDouble mean = new MatOfDouble();
//            MatOfDouble stddev = new MatOfDouble();
//            Core.meanStdDev(laplacian, mean, stddev);
//            double variance = stddev.get(0, 0)[0];
//            variance = variance * variance;
//
//            // ✅ Edge density check (for occlusion cases like fingers)
//            Imgproc.Canny(gray, edges, 50, 150);
//            double edgePixels = Core.countNonZero(edges);
//            double totalPixels = gray.rows() * gray.cols();
//            double edgeDensity = edgePixels / totalPixels;
//
//            // ✅ Multi-criteria decision
//            boolean isBlurred = variance < STRICT_BLUR_THRESHOLD || edgeDensity < EDGE_DENSITY_THRESHOLD;
//
//            // ✅ Confidence (scaled by both variance + edge density)
//            double sharpnessScore = Math.min(1.0, variance / (STRICT_BLUR_THRESHOLD * 2));
//            double edgeScore = Math.min(1.0, edgeDensity / (EDGE_DENSITY_THRESHOLD * 2));
//            double confidence = (sharpnessScore * 0.6 + edgeScore * 0.4) * 100.0;
//
//            String desc = getDescription(variance, edgeDensity, isBlurred);
//
//            Log.d(TAG, String.format("Blur Analysis - Variance: %.1f, EdgeDensity: %.3f, Blurred: %b, Confidence: %.1f%%, Desc: %s",
//                    variance, edgeDensity, isBlurred, confidence, desc));
//
//            return new BlurDetectionResult(variance, edgeDensity, isBlurred, desc, confidence);
//
//        } finally {
//            resized.release();
//            gray.release();
//            laplacian.release();
//            edges.release();
//        }
//    }
//
//    private static String getDescription(double variance, double edgeDensity, boolean blurred) {
//        if (blurred) {
//            if (edgeDensity < EDGE_DENSITY_THRESHOLD * 0.5) {
//                return "Too few details (possible obstruction)";
//            } else if (variance < STRICT_BLUR_THRESHOLD * 0.3) {
//                return "Severely blurred";
//            } else {
//                return "Blurred / low detail";
//            }
//        } else {
//            if (variance > STRICT_BLUR_THRESHOLD * 2) {
//                return "Very sharp";
//            } else {
//                return "Acceptably sharp";
//            }
//        }
//    }

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

                if (texture < TEXTURE_THRESHOLD) {
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
