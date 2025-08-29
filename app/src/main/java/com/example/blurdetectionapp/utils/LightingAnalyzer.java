package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class LightingAnalyzer {

    private static final String TAG = "LightingAnalyzer";

    // Thresholds - adjusted empirically
    private static final double SATURATION_THRESHOLD_BAD = 0.18;    // 18% saturation bad
    private static final double SATURATION_THRESHOLD_GOOD = 0.12;   // 12% saturation warning

    private static final double BRIGHT_PIXEL_RATIO_BAD = 0.3;       // 50% very bright pixels bad
    private static final double BRIGHT_PIXEL_RATIO_GOOD = 0.15;     // 35% bright pixels warning

    private static final double LAPLACIAN_VAR_THRESHOLD_BAD = 80.0; // below variance is bad
    private static final double LAPLACIAN_VAR_THRESHOLD_GOOD = 120.0;

    private static final int REFLECTION_MIN_AREA = 80;             // larger reflection area threshold
    private static final int REFLECTION_BRIGHTNESS_THRESHOLD = 230; // very bright pixels for reflection

    public enum LightingCondition {
        BAD, GOOD, PERFECT
    }

    public static class LightingAnalysisResult {
        public final double pixelSaturationRatio;
        public final double brightPixelRatio;
        public final double laplacianVariance;
        public final boolean hasReflection;
        public final LightingCondition lightingCondition;
        public final String statusMessage;
        public final String detailMessage;
        public final boolean isCaptureEnabled;

        public LightingAnalysisResult(double saturationRatio, double brightRatio,
                                      double lapVar, boolean reflection,
                                      LightingCondition condition,
                                      String status, String details, boolean captureEnabled) {
            this.pixelSaturationRatio = saturationRatio;
            this.brightPixelRatio = brightRatio;
            this.laplacianVariance = lapVar;
            this.hasReflection = reflection;
            this.lightingCondition = condition;
            this.statusMessage = status;
            this.detailMessage = details;
            this.isCaptureEnabled = captureEnabled;
        }
    }

    public static LightingAnalysisResult analyzeLighting(Bitmap bitmap) {
        if (bitmap == null) {
            return new LightingAnalysisResult(0, 0, 0, false,
                    LightingCondition.BAD, "No Image", "No image to analyze", false);
        }

        try {
            // Resize image for speed and noise reduction (optional)
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 800, 800 * bitmap.getHeight() / bitmap.getWidth(), true);

            Mat src = new Mat();
            Utils.bitmapToMat(resizedBitmap, src);

            // Convert to RGB
            Mat rgbMat = new Mat();
            Imgproc.cvtColor(src, rgbMat, Imgproc.COLOR_BGRA2RGB);

            double saturationRatio = analyzePixelSaturation(rgbMat);
            double brightRatio = analyzeBrightPixels(rgbMat);
            double laplacianVariance = analyzeLaplacianVariance(rgbMat);
            boolean reflectionDetected = detectReflection(rgbMat);

            LightingCondition condition = evaluateLightingCondition(saturationRatio, brightRatio, laplacianVariance, reflectionDetected);

            String status = generateStatusMessage(condition, saturationRatio, brightRatio, laplacianVariance, reflectionDetected);
            String details = generateDetailMessage(saturationRatio, brightRatio, laplacianVariance, reflectionDetected);

            boolean enableCapture = (condition == LightingCondition.GOOD || condition == LightingCondition.PERFECT);

            Log.d(TAG, String.format("Lighting - Sat: %.3f, Bright: %.3f, LapVar: %.1f, Refl: %b, Condition: %s",
                    saturationRatio, brightRatio, laplacianVariance, reflectionDetected, condition));

            return new LightingAnalysisResult(saturationRatio, brightRatio, laplacianVariance, reflectionDetected,
                    condition, status, details, enableCapture);

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing lighting", e);
            return new LightingAnalysisResult(0, 0, 0, false,
                    LightingCondition.BAD, "Analysis Error", "Failed to analyze lighting", false);
        }
    }

    // Saturation check using HSV
    private static double analyzePixelSaturation(Mat rgb) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);

        // Extract S channel
        List<Mat> channels = new ArrayList<>();
        Core.split(hsv, channels);
        Mat saturation = channels.get(1);

        int totalPixels = (int) saturation.total();

        // Count highly saturated pixels (S > 200 out of 255)
        Mat saturatedMask = new Mat();
        Imgproc.threshold(saturation, saturatedMask, 200, 255, Imgproc.THRESH_BINARY);

        int saturatedCount = Core.countNonZero(saturatedMask);

        return (double) saturatedCount / totalPixels;
    }

    // Bright pixel ratio from grayscale histogram
    private static double analyzeBrightPixels(Mat rgb) {
        Mat gray = new Mat();
        Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);

        // Compute histogram
        Mat hist = new Mat();
        Imgproc.calcHist(
                List.of(gray),
                new MatOfInt(0),
                new Mat(),
                hist,
                new MatOfInt(256),
                new MatOfFloat(0f, 256f));

        // Normalize histogram
        Core.normalize(hist, hist, 1, 0, Core.NORM_L1);

        double brightSum = 0;
        for (int i = 180; i < 256; i++) {
            double binVal = hist.get(i, 0)[0];
            brightSum += binVal;
        }
        return brightSum; // normalized sum, between 0 and 1
    }

    // Laplacian variance: higher means sharper, better contrast
    private static double analyzeLaplacianVariance(Mat rgb) {
        Mat gray = new Mat();
        Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);

        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(laplacian, mean, stddev);

        double variance = stddev.get(0,0)[0] * stddev.get(0,0)[0];
        return variance;
    }

    // Reflection detection using brightness threshold + morphology
    private static boolean detectReflection(Mat rgb) {
        Mat gray = new Mat();
        Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);

        // High brightness mask
        Mat brightMask = new Mat();
        Imgproc.threshold(gray, brightMask, REFLECTION_BRIGHTNESS_THRESHOLD, 255, Imgproc.THRESH_BINARY);

        // Morphological opening to remove noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7,7));
        Imgproc.morphologyEx(brightMask, brightMask, Imgproc.MORPH_OPEN, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(brightMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > REFLECTION_MIN_AREA) {
                // Confirm brightness inside contour
                Rect rect = Imgproc.boundingRect(contour);
                if (rect.x >= 0 && rect.y >= 0 &&
                        rect.x + rect.width <= gray.cols() &&
                        rect.y + rect.height <= gray.rows()) {
                    Mat roi = gray.submat(rect);
                    Scalar meanIntensity = Core.mean(roi);
                    if (meanIntensity.val[0] >= REFLECTION_BRIGHTNESS_THRESHOLD) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

//    private static boolean detectReflection(Mat rgb) {
//        Mat hsv = new Mat();
//        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);
//
//        // Split HSV channels
//        List<Mat> hsvChannels = new ArrayList<>();
//        Core.split(hsv, hsvChannels);
//        Mat vChannel = hsvChannels.get(2); // Value channel (brightness)
//        Mat sChannel = hsvChannels.get(1); // Saturation channel
//
//        // Threshold low saturation and high brightness pixels (candidate reflection)
//        Mat lowSaturationMask = new Mat();
//        Imgproc.threshold(sChannel, lowSaturationMask, 50, 255, Imgproc.THRESH_BINARY_INV);  // S < 50
//
//        Mat highBrightnessMask = new Mat();
//        Imgproc.threshold(vChannel, highBrightnessMask, 220, 255, Imgproc.THRESH_BINARY); // V > 220 (tuned threshold)
//
//        Mat reflectionCandidates = new Mat();
//        Core.bitwise_and(lowSaturationMask, highBrightnessMask, reflectionCandidates);
//
//        // Morphological opening and closing to remove noise and fill gaps
//        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5));
//        Imgproc.morphologyEx(reflectionCandidates, reflectionCandidates, Imgproc.MORPH_OPEN, kernel);
//        Imgproc.morphologyEx(reflectionCandidates, reflectionCandidates, Imgproc.MORPH_CLOSE, kernel);
//
//        // Find contours in reflection candidate mask
//        List<MatOfPoint> contours = new ArrayList<>();
//        Mat hierarchy = new Mat();
//        Imgproc.findContours(reflectionCandidates, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//
//        for (MatOfPoint contour : contours) {
//            double area = Imgproc.contourArea(contour);
//            if (area > 50) {  // Smaller area threshold
//                Rect rect = Imgproc.boundingRect(contour);
//                // Validate brightness again in the candidate region
//                Mat roi = vChannel.submat(rect);
//                Scalar meanBrightness = Core.mean(roi);
//                if (meanBrightness.val[0] > 220) {
//                    // Potential reflection detected
//                    return true;
//                }
//            }
//        }
//
//        return false;
//    }


    private static LightingCondition evaluateLightingCondition(double saturationRatio, double brightRatio,
                                                               double laplacianVar, boolean hasReflection) {
        if (hasReflection) {
            return LightingCondition.BAD;
        }
        if (saturationRatio > SATURATION_THRESHOLD_BAD
                || brightRatio > BRIGHT_PIXEL_RATIO_BAD
                || laplacianVar < LAPLACIAN_VAR_THRESHOLD_BAD) {
            return LightingCondition.BAD;
        }
        if (saturationRatio > SATURATION_THRESHOLD_GOOD
                || brightRatio > BRIGHT_PIXEL_RATIO_GOOD
                || laplacianVar < LAPLACIAN_VAR_THRESHOLD_GOOD) {
            return LightingCondition.GOOD;
        }
        return LightingCondition.PERFECT;
    }

    private static String generateStatusMessage(LightingCondition condition, double saturationRatio,
                                                double brightRatio, double laplacianVar, boolean hasReflection) {
        switch (condition) {
            case PERFECT:
                return "✓ Perfect Lighting";
            case GOOD:
                StringBuilder warning = new StringBuilder("⚠ Warning: ");
                if (saturationRatio > SATURATION_THRESHOLD_GOOD) warning.append("High Saturation; ");
                if (brightRatio > BRIGHT_PIXEL_RATIO_GOOD) warning.append("Too Bright; ");
                if (laplacianVar < LAPLACIAN_VAR_THRESHOLD_GOOD) warning.append("Low Contrast; ");
                if (hasReflection) warning.append("Reflection Detected; ");
                if (warning.length() > 9) warning.setLength(warning.length() - 2);
                else warning.append("Minor Issues");
                return warning.toString();
            case BAD:
            default:
                if (hasReflection) return "✗ Bad Lighting: Reflection Detected";
                if (saturationRatio > SATURATION_THRESHOLD_BAD) return "✗ Bad Lighting: High Saturation";
                if (brightRatio > BRIGHT_PIXEL_RATIO_BAD) return "✗ Bad Lighting: Too Bright";
                if (laplacianVar < LAPLACIAN_VAR_THRESHOLD_BAD) return "✗ Bad Lighting: Low Contrast";
                return "✗ Bad Lighting";
        }
    }

    private static String generateDetailMessage(double saturationRatio, double brightRatio, double laplacianVar, boolean hasReflection) {
        return String.format("Saturation: %.1f%% | Brightness: %.1f%% | Contrast: %.1f | Reflection: %s",
                saturationRatio * 100, brightRatio * 100, laplacianVar, hasReflection ? "Yes" : "No");
    }
}
