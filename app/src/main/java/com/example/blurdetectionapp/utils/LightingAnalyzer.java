package com.example.blurdetectionapp.utils;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class LightingAnalyzer {

    private static final String TAG = "LightingAnalyzer";

    // Thresholds
    private static final double BRIGHT_PIXEL_RATIO_BAD = 0.35;   // % of very bright pixels considered too much
    private static final double BRIGHT_PIXEL_RATIO_WARN = 0.20;  // warning if moderately bright

    private static final int REFLECTION_MIN_AREA = 150;          // area of bright blob to be counted as reflection
    private static final int REFLECTION_BRIGHTNESS_THRESHOLD = 240; // pixel threshold for reflection

    // Processing resize for speed
    private static final int PROCESSING_WIDTH = 640;

    public enum LightingCondition {
        BAD, GOOD, PERFECT
    }

    public static class LightingAnalysisResult {
        public final double brightPixelRatio;
        public final boolean hasReflection;
        public final LightingCondition lightingCondition;
        public final String statusMessage;
        public final String detailMessage;
        public final boolean isCaptureEnabled;

        public LightingAnalysisResult(double brightRatio, boolean reflection,
                                      LightingCondition condition, String status, String details, boolean captureEnabled) {
            this.brightPixelRatio = brightRatio;
            this.hasReflection = reflection;
            this.lightingCondition = condition;
            this.statusMessage = status;
            this.detailMessage = details;
            this.isCaptureEnabled = captureEnabled;
        }
    }

    /**
     * Main entry - analyze lighting from input Mat (gray or BGR).
     */
    public static LightingAnalysisResult analyzeLighting(Mat input) {
        if (input == null || input.empty()) {
            return new LightingAnalysisResult(0, false,
                    LightingCondition.BAD, "No Image", "No frame to analyze", false);
        }

        try {
            // Ensure grayscale
            Mat gray = new Mat();
            if (input.channels() > 1) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Resize for faster processing
            double aspectRatio = (double) gray.height() / gray.width();
            int newHeight = (int) (PROCESSING_WIDTH * aspectRatio);
            Mat resized = new Mat();
            Imgproc.resize(gray, resized, new Size(PROCESSING_WIDTH, newHeight));

            // Brightness ratio
            double brightRatio = analyzeBrightPixels(resized);

            // Reflection detection
            boolean reflectionDetected = detectReflection(resized);

            // Evaluate
            LightingCondition condition = evaluateLightingCondition(brightRatio, reflectionDetected);

            String status = generateStatusMessage(condition, brightRatio, reflectionDetected);
            String details = generateDetailMessage(brightRatio, reflectionDetected);

            boolean enableCapture = (condition == LightingCondition.GOOD || condition == LightingCondition.PERFECT);

            Log.d(TAG, String.format("Lighting - Bright: %.3f, Refl: %b, Condition: %s",
                    brightRatio, reflectionDetected, condition));

            gray.release();
            resized.release();

            return new LightingAnalysisResult(brightRatio, reflectionDetected,
                    condition, status, details, enableCapture);

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing lighting", e);
            return new LightingAnalysisResult(0, false,
                    LightingCondition.BAD, "Analysis Error", "Failed to analyze lighting", false);
        }
    }

    // Count bright pixels above a threshold
    private static double analyzeBrightPixels(Mat gray) {
        Mat thresh = new Mat();
        Imgproc.threshold(gray, thresh, 200, 255, Imgproc.THRESH_BINARY);

        double brightPixelRatio = (double) Core.countNonZero(thresh) / (gray.rows() * gray.cols());

        thresh.release();
        return brightPixelRatio;
    }

    // Detect large bright regions (reflections)
    private static boolean detectReflection(Mat gray) {
        Mat brightMask = new Mat();
        Imgproc.threshold(gray, brightMask, REFLECTION_BRIGHTNESS_THRESHOLD, 255, Imgproc.THRESH_BINARY);

        // Morph open to remove small dots
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(brightMask, brightMask, Imgproc.MORPH_OPEN, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(brightMask, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        boolean hasReflection = false;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > REFLECTION_MIN_AREA) {
                hasReflection = true;
                break;
            }
        }

        brightMask.release();
        kernel.release();
        hierarchy.release();
        for (MatOfPoint contour : contours) {
            contour.release();
        }

        return hasReflection;
    }

    private static LightingCondition evaluateLightingCondition(double brightRatio, boolean hasReflection) {
        if (hasReflection) return LightingCondition.BAD;
        if (brightRatio > BRIGHT_PIXEL_RATIO_BAD) return LightingCondition.BAD;
        if (brightRatio > BRIGHT_PIXEL_RATIO_WARN) return LightingCondition.GOOD;
        return LightingCondition.PERFECT;
    }

    private static String generateStatusMessage(LightingCondition condition, double brightRatio, boolean hasReflection) {
        switch (condition) {
            case PERFECT:
                return "✓ Perfect Lighting";
            case GOOD:
                return "⚠ Good Lighting - Slightly Bright";
            case BAD:
            default:
                if (hasReflection) return "✗ Bad Lighting: Reflection Detected";
                if (brightRatio > BRIGHT_PIXEL_RATIO_BAD) return "✗ Bad Lighting: Too Bright";
                return "✗ Bad Lighting";
        }
    }

    private static String generateDetailMessage(double brightRatio, boolean hasReflection) {
        return String.format("Brightness: %.1f%% | Reflection: %s",
                brightRatio * 100, hasReflection ? "Yes" : "No");
    }
}
