package com.example.blurdetectionapp.utils;

import android.annotation.SuppressLint;
import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class LightingAnalyzer {

    private static final String TAG = "LightingAnalyzer";

    // Relaxed thresholds for less sensitivity
    private static final double BRIGHT_PIXEL_RATIO_BAD = 0.4;       // 40% very bright pixels bad
    private static final double BRIGHT_PIXEL_RATIO_GOOD = 0.25;     // 25% bright pixels warning

    private static final double LAPLACIAN_VAR_THRESHOLD_BAD = 50.0; // Lower threshold - less strict
    private static final double LAPLACIAN_VAR_THRESHOLD_GOOD = 80.0;

    private static final int REFLECTION_MIN_AREA = 150;             // Larger area to avoid false positives
    private static final int REFLECTION_BRIGHTNESS_THRESHOLD = 240; // Higher threshold for reflections

    // Processing size for speed optimization
    private static final int PROCESSING_WIDTH = 640;

    public enum LightingCondition {
        BAD, GOOD, PERFECT
    }

    public static class LightingAnalysisResult {
        public final double brightPixelRatio;
        public final double laplacianVariance;
        public final boolean hasReflection;
        public final LightingCondition lightingCondition;
        public final String statusMessage;
        public final String detailMessage;
        public final boolean isCaptureEnabled;

        public LightingAnalysisResult(double brightRatio, double lapVar, boolean reflection,
                                      LightingCondition condition, String status, String details, boolean captureEnabled) {
            this.brightPixelRatio = brightRatio;
            this.laplacianVariance = lapVar;
            this.hasReflection = reflection;
            this.lightingCondition = condition;
            this.statusMessage = status;
            this.detailMessage = details;
            this.isCaptureEnabled = captureEnabled;
        }
    }

    public static LightingAnalysisResult analyzeLighting(Mat src) {
        if (src == null || src.empty()) {
            return new LightingAnalysisResult(0, 0, false,
                    LightingCondition.BAD, "No Image", "No image to analyze", false);
        }

        try {
            // Resize for faster processing - maintain aspect ratio
            int originalWidth = src.width();
            int originalHeight = src.height();
            int newHeight = (PROCESSING_WIDTH * originalHeight) / originalWidth;

            Mat resized = new Mat();
            Size newSize = new Size(PROCESSING_WIDTH, newHeight);
            Imgproc.resize(src, resized, newSize);

            // Convert to grayscale once for all operations
            Mat gray = new Mat();
            if (resized.channels() == 3) {
                Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);
            } else if (resized.channels() == 4) {
                Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGRA2GRAY);
            } else {
                gray = resized.clone();
            }

            // Analyze only essential metrics
            double brightRatio = analyzeBrightPixelsOptimized(gray);
            double laplacianVariance = analyzeLaplacianVariance(gray);
            boolean reflectionDetected = detectReflectionOptimized(gray);

            LightingCondition condition = evaluateLightingCondition(brightRatio, laplacianVariance, reflectionDetected);

            String status = generateStatusMessage(condition, brightRatio, laplacianVariance, reflectionDetected);
            String details = generateDetailMessage(brightRatio, laplacianVariance, reflectionDetected);

            boolean enableCapture = (condition == LightingCondition.GOOD || condition == LightingCondition.PERFECT);

            Log.d(TAG, String.format("Lighting - Bright: %.3f, LapVar: %.1f, Refl: %b, Condition: %s",
                    brightRatio, laplacianVariance, reflectionDetected, condition));

            // Clean up
            resized.release();
            gray.release();

            return new LightingAnalysisResult(brightRatio, laplacianVariance, reflectionDetected,
                    condition, status, details, enableCapture);

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing lighting", e);
            return new LightingAnalysisResult(0, 0, false,
                    LightingCondition.BAD, "Analysis Error", "Failed to analyze lighting", false);
        }
    }

    // Optimized bright pixel analysis - direct threshold on grayscale
    private static double analyzeBrightPixelsOptimized(Mat gray) {
        int totalPixels = (int) gray.total();

        // Count pixels above brightness threshold (200 out of 255)
        Mat brightMask = new Mat();
        Imgproc.threshold(gray, brightMask, 200, 255, Imgproc.THRESH_BINARY);

        int brightCount = Core.countNonZero(brightMask);
        brightMask.release();

        return (double) brightCount / totalPixels;
    }

    // Optimized Laplacian variance calculation
    private static double analyzeLaplacianVariance(Mat gray) {
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(laplacian, mean, stddev);

        double variance = stddev.get(0, 0)[0] * stddev.get(0, 0)[0];

        laplacian.release();
        return variance;
    }

    // Simplified reflection detection - focus on large bright areas
    private static boolean detectReflectionOptimized(Mat gray) {
        // High brightness mask with stricter threshold
        Mat brightMask = new Mat();
        Imgproc.threshold(gray, brightMask, REFLECTION_BRIGHTNESS_THRESHOLD, 255, Imgproc.THRESH_BINARY);

        // Morphological opening to remove small noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(brightMask, brightMask, Imgproc.MORPH_OPEN, kernel);

        // Find large contours only
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(brightMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        boolean hasReflection = false;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > REFLECTION_MIN_AREA) {
                hasReflection = true;
                break; // Found one large bright area, that's enough
            }
        }

        // Clean up
        brightMask.release();
        kernel.release();
        hierarchy.release();
        for (MatOfPoint contour : contours) {
            contour.release();
        }

        return hasReflection;
    }

    private static LightingCondition evaluateLightingCondition(double brightRatio, double laplacianVar, boolean hasReflection) {
        // Reflection is always bad
        if (hasReflection) {
            return LightingCondition.BAD;
        }

        // Check for bad conditions
        if (brightRatio > BRIGHT_PIXEL_RATIO_BAD || laplacianVar < LAPLACIAN_VAR_THRESHOLD_BAD) {
            return LightingCondition.BAD;
        }

        // Check for good conditions
        if (brightRatio > BRIGHT_PIXEL_RATIO_GOOD || laplacianVar < LAPLACIAN_VAR_THRESHOLD_GOOD) {
            return LightingCondition.GOOD;
        }

        return LightingCondition.PERFECT;
    }

    private static String generateStatusMessage(LightingCondition condition, double brightRatio,
                                                double laplacianVar, boolean hasReflection) {
        switch (condition) {
            case PERFECT:
                return "✓ Perfect Lighting";
            case GOOD:
                StringBuilder warning = new StringBuilder("⚠ Good Lighting");
                if (brightRatio > BRIGHT_PIXEL_RATIO_GOOD) warning.append(" - Slightly Bright");
                if (laplacianVar < LAPLACIAN_VAR_THRESHOLD_GOOD) warning.append(" - Low Contrast");
                return warning.toString();
            case BAD:
            default:
                if (hasReflection) return "✗ Bad Lighting: Reflection Detected";
                if (brightRatio > BRIGHT_PIXEL_RATIO_BAD) return "✗ Bad Lighting: Too Bright";
                if (laplacianVar < LAPLACIAN_VAR_THRESHOLD_BAD) return "✗ Bad Lighting: Very Low Contrast";
                return "✗ Bad Lighting";
        }
    }

    @SuppressLint("DefaultLocale")
    private static String generateDetailMessage(double brightRatio, double laplacianVar, boolean hasReflection) {
        return String.format("Brightness: %.1f%% | Contrast: %.1f | Reflection: %s",
                brightRatio * 100, laplacianVar, hasReflection ? "Yes" : "No");
    }
}
