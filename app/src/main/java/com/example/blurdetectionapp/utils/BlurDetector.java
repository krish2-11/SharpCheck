package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

/**
 * Utility class for detecting blur in images using Laplacian variance method
 * Optimized for detecting subtle blur in real-world scenarios like partial defocus.
 */
public class BlurDetector {

    private static final String TAG = "BlurDetector";

    // Different thresholds for different sensitivity levels
    public static final double STRICT_BLUR_THRESHOLD = 600.0;    // More sensitive - catches subtle blur
    public static final double MODERATE_BLUR_THRESHOLD = 800.0;  // Balanced approach
    public static final double LENIENT_BLUR_THRESHOLD = 1200.0;  // Less sensitive - only obvious blur

    // Default threshold - recommended for your use case
    private static final double DEFAULT_BLUR_THRESHOLD = STRICT_BLUR_THRESHOLD;

    // Resize dimensions for faster processing
    private static final int RESIZE_WIDTH = 320;

    public static class BlurDetectionResult {
        public final double laplacianVariance;
        public final boolean isBlurred;
        public final String description;
        public final double confidenceScore; // 0-100, higher = more confident it's sharp

        public BlurDetectionResult(double variance, boolean blurred, String desc, double confidence) {
            this.laplacianVariance = variance;
            this.isBlurred = blurred;
            this.description = desc;
            this.confidenceScore = confidence;
        }
    }

    /**
     * Detect blur with default threshold (strict)
     */
    public static BlurDetectionResult detectBlur(Bitmap bitmap) {
        return detectBlur(bitmap, DEFAULT_BLUR_THRESHOLD);
    }

    /**
     * Detect blur with custom threshold
     *
     * @param bitmap The image to analyze
     * @param threshold Custom blur threshold
     * @return BlurDetectionResult containing analysis results
     */
    public static BlurDetectionResult detectBlur(Bitmap bitmap, double threshold) {
        if (bitmap == null) {
            return new BlurDetectionResult(0, true, "No image provided", 0);
        }

        Mat mat = null;
        Mat gray = null;
        Mat laplacian = null;

        try {
            // Resize bitmap for faster processing
            int originalWidth = bitmap.getWidth();
            int originalHeight = bitmap.getHeight();
            int resizedHeight = (int) ((float) RESIZE_WIDTH / originalWidth * originalHeight);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, RESIZE_WIDTH, resizedHeight, false);

            // Convert Bitmap to Mat
            mat = new Mat();
            Utils.bitmapToMat(resizedBitmap, mat);

            // Convert to Grayscale
            gray = new Mat();
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY);

            // Apply Laplacian operator
            laplacian = new Mat();
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

            // Calculate variance of Laplacian
            MatOfDouble mean = new MatOfDouble();
            MatOfDouble stddev = new MatOfDouble();
            org.opencv.core.Core.meanStdDev(laplacian, mean, stddev);

            double variance = stddev.get(0, 0)[0];
            variance = variance * variance; // variance = stddev^2

            boolean isBlurred = variance < threshold;

            // Calculate confidence score (0-100)
            double confidenceScore = Math.min(100, (variance / threshold) * 100);

            String description = getBlurDescription(variance, threshold);

            Log.d(TAG, String.format("Blur Analysis - Variance: %.1f, Threshold: %.1f, Blurred: %b, " +
                            "Confidence: %.1f%%, Description: %s",
                    variance, threshold, isBlurred, confidenceScore, description));

            return new BlurDetectionResult(variance, isBlurred, description, confidenceScore);

        } catch (Exception e) {
            Log.e(TAG, "Error detecting blur", e);
            return new BlurDetectionResult(0, true, "Analysis failed", 0);
        } finally {
            // Release native memory
            if (mat != null) mat.release();
            if (gray != null) gray.release();
            if (laplacian != null) laplacian.release();
        }
    }

    /**
     * Get descriptive blur assessment based on variance and threshold
     */
    private static String getBlurDescription(double variance, double threshold) {
        double ratio = variance / threshold;

        if (variance < threshold * 0.3) {
            return "Severely Blurred";
        } else if (variance < threshold * 0.6) {
            return "Moderately Blurred";
        } else if (variance < threshold) {
            return "Slightly Blurred";
        } else if (variance < threshold * 1.5) {
            return "Acceptable Sharpness";
        } else if (variance < threshold * 2.5) {
            return "Sharp";
        } else {
            return "Very Sharp";
        }
    }

    /**
     * Detect blur with different sensitivity levels
     */
    public static BlurDetectionResult detectBlurStrict(Bitmap bitmap) {
        return detectBlur(bitmap, STRICT_BLUR_THRESHOLD);
    }

    public static BlurDetectionResult detectBlurModerate(Bitmap bitmap) {
        return detectBlur(bitmap, MODERATE_BLUR_THRESHOLD);
    }

    public static BlurDetectionResult detectBlurLenient(Bitmap bitmap) {
        return detectBlur(bitmap, LENIENT_BLUR_THRESHOLD);
    }

    /**
     * Simple boolean check (backward compatibility)
     */
    public static boolean isImageBlurred(Bitmap bitmap) {
        return detectBlur(bitmap).isBlurred;
    }

    /**
     * Check if image has acceptable quality for your specific use case
     * Considers both blur and confidence level
     */
    public static boolean isImageAcceptableForCapture(Bitmap bitmap) {
        BlurDetectionResult result = detectBlurModerate(bitmap);
        // Accept if not blurred OR if confidence is above 70%
        return !result.isBlurred || result.confidenceScore > 70;
    }

    /**
     * Get a quality assessment score (0-100) where higher is better
     */
    public static double getImageQualityScore(Bitmap bitmap) {
        BlurDetectionResult result = detectBlur(bitmap);
        return result.confidenceScore;
    }
}