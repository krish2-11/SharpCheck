package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class ShadowDetection {

    /**
     * Detect shadows using adaptive thresholding on Value channel in HSV
     * Returns a mask Bitmap where shadows are white
     */
    public static Bitmap detectShadows(Bitmap inputBitmap) {
        if (inputBitmap == null) return null;

        // Convert Bitmap to Mat
        Mat src = new Mat();
        Utils.bitmapToMat(inputBitmap, src);

        // Convert to HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_RGB2HSV);

        // Extract Value channel
        java.util.List<Mat> channels = new java.util.ArrayList<>();
        Core.split(hsv, channels);
        Mat v = channels.get(2);

        // Apply Gaussian blur to reduce noise
        Imgproc.GaussianBlur(v, v, new Size(5, 5), 0);

        // Adaptive threshold to detect dark regions (shadows)
        Mat shadowMask = new Mat();
        Imgproc.adaptiveThreshold(v, shadowMask, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV, 25, 10);

        // Optional: remove small noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_CLOSE, kernel);

        // Optional: overlay red on shadows for visualization
        Mat redOverlay = new Mat(src.size(), src.type(), new Scalar(255, 0, 0));
        Mat shadowOverlay = new Mat();
        redOverlay.copyTo(shadowOverlay, shadowMask);

        Mat result = new Mat();
        Core.addWeighted(src, 1.0, shadowOverlay, 0.5, 0, result);

        // Convert mask for output
        Bitmap outputBitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, outputBitmap);

        // Release mats
        src.release();
        hsv.release();
        v.release();
        shadowMask.release();
        kernel.release();
        redOverlay.release();
        shadowOverlay.release();
        result.release();

        return outputBitmap;
    }
}
