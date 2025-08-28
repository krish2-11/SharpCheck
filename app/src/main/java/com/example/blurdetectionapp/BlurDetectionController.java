package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import android.util.Log;
import java.util.List;

/**
 * Main controller that orchestrates blur detection and deblurring process
 */
public class BlurDetectionController {
    private static final String TAG = "BlurDetectionController";

    private BlurredObjectDetector blurredObjectDetector;
    private ImageDeblurrer imageDeblurrer;

    public BlurDetectionController() {
        this.blurredObjectDetector = new BlurredObjectDetector();
        this.imageDeblurrer = new ImageDeblurrer();
    }

    /**
     * Complete pipeline: detect blurred objects, deblur them, and return processed image
     */
    public BlurDetectionControllerResult processImage(Bitmap inputBitmap) {
        Log.d(TAG, "Starting blur detection and deblurring pipeline");

        try {
            // Step 1: Detect blurred objects
            Log.d(TAG, "Step 1: Detecting blurred objects");
            BlurredObjectDetectionResult detectionResult =
                    blurredObjectDetector.detectBlurredObjects(inputBitmap);

            if (!detectionResult.hasBlurredObjects()) {
                Log.d(TAG, "No blurred objects detected, returning original image");
                return new BlurDetectionControllerResult(
                        inputBitmap,
                        detectionResult,
                        null,
                        true,
                        "No blurred objects detected - image is ready for object detection"
                );
            }

            Log.d(TAG, "Found " + detectionResult.getBlurredObjectCount() +
                    " blurred objects out of " + detectionResult.getTotalObjectsDetected() + " total objects");

            // Step 2: Deblur the detected objects
            Log.d(TAG, "Step 2: Deblurring detected objects");
            DeblurringResult deblurringResult =
                    imageDeblurrer.deblurImage(inputBitmap, detectionResult.getBlurredObjects());

            if (!deblurringResult.isSuccess()) {
                Log.w(TAG, "Deblurring failed, returning original image");
                return new BlurDetectionControllerResult(
                        inputBitmap,
                        detectionResult,
                        deblurringResult,
                        false,
                        "Deblurring failed - using original image for object detection"
                );
            }

            // Step 3: Return processed image
            Log.d(TAG, "Pipeline completed successfully");
            String message = String.format(
                    "Successfully processed %d blurred objects. Average quality improvement: %.1f%%",
                    deblurringResult.getDeblurredObjectCount(),
                    calculateAverageQualityImprovement(deblurringResult.getDeblurredObjects())
            );

            return new BlurDetectionControllerResult(
                    deblurringResult.getProcessedImage(),
                    detectionResult,
                    deblurringResult,
                    true,
                    message
            );

        } catch (Exception e) {
            Log.e(TAG, "Error in blur detection pipeline", e);
            return new BlurDetectionControllerResult(
                    inputBitmap,
                    null,
                    null,
                    false,
                    "Pipeline failed: " + e.getMessage()
            );
        }
    }

    /**
     * Quick check to see if image has blurred objects (without deblurring)
     */
    public boolean hasBlurredObjects(Bitmap inputBitmap) {
        try {
            BlurredObjectDetectionResult result = blurredObjectDetector.detectBlurredObjects(inputBitmap);
            return result.hasBlurredObjects();
        } catch (Exception e) {
            Log.e(TAG, "Error checking for blurred objects", e);
            return false;
        }
    }

    /**
     * Get detailed blur analysis without deblurring
     */
    public BlurAnalysisResult analyzeBlur(Bitmap inputBitmap) {
        try {
            BlurredObjectDetectionResult detectionResult =
                    blurredObjectDetector.detectBlurredObjects(inputBitmap);

            return new BlurAnalysisResult(
                    detectionResult.getTotalObjectsDetected(),
                    detectionResult.getBlurredObjectCount(),
                    calculateBlurSeverity(detectionResult.getBlurredObjects()),
                    detectionResult.hasBlurredObjects()
            );

        } catch (Exception e) {
            Log.e(TAG, "Error in blur analysis", e);
            return new BlurAnalysisResult(0, 0, 0.0, false);
        }
    }

    private double calculateAverageQualityImprovement(List<DeblurredObject> deblurredObjects) {
        if (deblurredObjects.isEmpty()) return 0.0;

        double total = 0.0;
        for (DeblurredObject obj : deblurredObjects) {
            total += obj.getQualityImprovement();
        }

        return total / deblurredObjects.size();
    }

    private double calculateBlurSeverity(List<BlurredObject> blurredObjects) {
        if (blurredObjects.isEmpty()) return 0.0;

        double totalSeverity = 0.0;
        for (BlurredObject obj : blurredObjects) {
            // Calculate severity based on Laplacian variance (lower = more blurred)
            double severity = Math.max(0, 100 - obj.getLaplacianVariance()) / 100.0;
            totalSeverity += severity;
        }

        return totalSeverity / blurredObjects.size();
    }
}

/**
 * Complete result of the blur detection and deblurring pipeline
 */
class BlurDetectionControllerResult {
    private Bitmap processedImage;
    private BlurredObjectDetectionResult detectionResult;
    private DeblurringResult deblurringResult;
    private boolean success;
    private String message;

    public BlurDetectionControllerResult(Bitmap processedImage,
                                         BlurredObjectDetectionResult detectionResult,
                                         DeblurringResult deblurringResult,
                                         boolean success,
                                         String message) {
        this.processedImage = processedImage;
        this.detectionResult = detectionResult;
        this.deblurringResult = deblurringResult;
        this.success = success;
        this.message = message;
    }

    // Getters
    public Bitmap getProcessedImage() { return processedImage; }
    public BlurredObjectDetectionResult getDetectionResult() { return detectionResult; }
    public DeblurringResult getDeblurringResult() { return deblurringResult; }
    public boolean isSuccess() { return success; }
    public String getMessage() { return message; }

    // Convenience methods
    public boolean hadBlurredObjects() {
        return detectionResult != null && detectionResult.hasBlurredObjects();
    }

    public boolean wasDeblurred() {
        return deblurringResult != null && deblurringResult.isSuccess();
    }

    public int getTotalObjectsDetected() {
        return detectionResult != null ? detectionResult.getTotalObjectsDetected() : 0;
    }

    public int getBlurredObjectsCount() {
        return detectionResult != null ? detectionResult.getBlurredObjectCount() : 0;
    }
}

/**
 * Result of blur analysis without deblurring
 */
class BlurAnalysisResult {
    private int totalObjects;
    private int blurredObjects;
    private double averageBlurSeverity;
    private boolean hasBlur;

    public BlurAnalysisResult(int totalObjects, int blurredObjects,
                              double averageBlurSeverity, boolean hasBlur) {
        this.totalObjects = totalObjects;
        this.blurredObjects = blurredObjects;
        this.averageBlurSeverity = averageBlurSeverity;
        this.hasBlur = hasBlur;
    }

    public int getTotalObjects() { return totalObjects; }
    public int getBlurredObjects() { return blurredObjects; }
    public double getAverageBlurSeverity() { return averageBlurSeverity; }
    public boolean hasBlur() { return hasBlur; }
    public double getBlurPercentage() {
        return totalObjects > 0 ? (double) blurredObjects / totalObjects * 100.0 : 0.0;
    }
}
