package com.example.blurdetectionapp;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ShadowDetection {

    /**
     * Detect shadows in a Bitmap and overlay red highlights on shadow regions.
     * This method uses HSV color space thresholds on Value and Saturation channels
     * to distinguish shadows from black or gray objects.
     *
     * @param inputBitmap input frame
     * @return bitmap with red overlay on detected shadows
     */
    public static Bitmap detectShadows(Bitmap inputBitmap) {
        if (inputBitmap == null) return null;

        // Convert Bitmap to Mat
        Mat src = new Mat();
        Utils.bitmapToMat(inputBitmap, src);

        // Convert to HSV color space
        Mat hsv = new Mat();
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_RGB2HSV);

        // Split HSV channels
        List<Mat> channels = new ArrayList<>();
        Core.split(hsv, channels);
        Mat hChannel = channels.get(0);
        Mat sChannel = channels.get(1);
        Mat vChannel = channels.get(2);

        // Threshold dark regions on Value channel (pixels with V < 50)
        Mat darkMask = new Mat();
        Imgproc.threshold(vChannel, darkMask, 50, 255, Imgproc.THRESH_BINARY_INV);

        // Threshold saturation to exclude very low saturation (black/gray areas)
        Mat satMask = new Mat();
        Imgproc.threshold(sChannel, satMask, 30, 255, Imgproc.THRESH_BINARY);

        // Combine masks: dark AND sufficiently saturated pixels
        Mat shadowMask = new Mat();
        Core.bitwise_and(darkMask, satMask, shadowMask);

        // Morphological operations to clean noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_CLOSE, kernel);

        // Create a red overlay Mat (same size and type as source)
        Mat redOverlay = new Mat(src.size(), src.type(), new Scalar(255, 0, 0));

        // Copy red overlay only to shadow regions using mask
        Mat shadowOverlay = new Mat();
        redOverlay.copyTo(shadowOverlay, shadowMask);

        // Blend original image and shadow overlay (50% opacity)
        Mat result = new Mat();
        Core.addWeighted(src, 1.0, shadowOverlay, 0.5, 0.0, result);

        // Convert result Mat back to Bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, outputBitmap);

        // Release Mats to free memory
        src.release();
        hsv.release();
        hChannel.release();
        sChannel.release();
        vChannel.release();
        darkMask.release();
        satMask.release();
        shadowMask.release();
        kernel.release();
        redOverlay.release();
        shadowOverlay.release();
        result.release();

        return outputBitmap;
    }
}