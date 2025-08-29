package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

/**
 * Utility class for detecting blur in images using Laplacian variance method
 */
public class BlurDetector {

    private static final String TAG = "BlurDetector";
    private static final double BLUR_THRESHOLD = 1500.0; // Variance threshold for blur detection

    /**
     * Result class for blur detection
     */
    public static class BlurDetectionResult {
        public final double laplacianVariance;
        public final boolean isBlurred;
        public final String description;

        public BlurDetectionResult(double variance, boolean blurred, String desc) {
            this.laplacianVariance = variance;
            this.isBlurred = blurred;
            this.description = desc;
        }
    }

    /**
     * Detect if the given bitmap is blurred
     * @param bitmap The image to analyze
     * @return BlurDetectionResult containing analysis results
     */
    public static BlurDetectionResult detectBlur(Bitmap bitmap) {
        if (bitmap == null) {
            return new BlurDetectionResult(0, true, "No image provided");
        }

        try {
            Mat mat = new Mat();
            Utils.bitmapToMat(bitmap, mat);

            Mat gray = new Mat();
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);

            Mat laplacian = new Mat();
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

            MatOfDouble mean = new MatOfDouble();
            MatOfDouble stddev = new MatOfDouble();
            Core.meanStdDev(laplacian, mean, stddev);

            double variance = stddev.get(0, 0)[0] * stddev.get(0, 0)[0];
            boolean isBlurred = variance < BLUR_THRESHOLD;

            String description;
            if (isBlurred) {
                if (variance < 500) {
                    description = "Severely Blurred";
                } else if (variance < 1000) {
                    description = "Moderately Blurred";
                } else {
                    description = "Slightly Blurred";
                }
            } else {
                if (variance > 3000) {
                    description = "Very Sharp";
                } else if (variance > 2000) {
                    description = "Sharp";
                } else {
                    description = "Acceptable Sharpness";
                }
            }

            Log.d(TAG, String.format("Blur Analysis - Variance: %.1f, Blurred: %b, Description: %s",
                    variance, isBlurred, description));

            return new BlurDetectionResult(variance, isBlurred, description);

        } catch (Exception e) {
            Log.e(TAG, "Error detecting blur", e);
            return new BlurDetectionResult(0, true, "Analysis failed");
        }
    }

    /**
     * Simple boolean check for blur (backward compatibility)
     * @param bitmap The image to analyze
     * @return true if image is blurred, false otherwise
     */
    public static boolean isImageBlurred(Bitmap bitmap) {
        BlurDetectionResult result = detectBlur(bitmap);
        return result.isBlurred;
    }
}