package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.*;

public class DocumentDetection {

    private static final String TAG = "DocumentDetection";
    private Point[] lastCorners = null;

    /**
     * Detects document corners and returns them as Point[]
     * Works on any darker background
     */
    public Point[] detectDocumentCornersPoints(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        // 1️⃣ Convert to grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);

         //Remove background by thresholding
        Mat mask = new Mat();
        Imgproc.threshold(
                grayMat,               // input gray image
                mask,                   // output binary mask
                0,                      // threshold value (ignored with Outs)
                255,                     // max value
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
        );
        // 3️⃣ Keep only the document (white areas in mask)
        Mat foreground = new Mat();
        mat.copyTo(foreground, mask);

        Imgproc.GaussianBlur(foreground, foreground, new Size(5, 5), 0);
        Mat dst = new Mat();

        // Define a sharpening kernel
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        float[] data = {
                0, -1,  0,
                -1,  5, -1,
                0, -1,  0
        };
        kernel.put(0, 0, data);

        // Apply the filter
        Imgproc.filter2D(foreground, dst, foreground.depth(), kernel);

        // 3️⃣ Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(dst, edges, 25, 150);

        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel2);

        // 5️⃣ Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Log.d(TAG, "Total contours found: " + contours.size());

        // 6️⃣ Sort contours by area (largest first)
        contours.sort((c1, c2) -> Double.compare(Imgproc.contourArea(c2), Imgproc.contourArea(c1)));

        int imgArea = mat.cols() * mat.rows();
        double minArea = imgArea * 0.1; // Ignore small contours

        MatOfPoint2f approxCurve = new MatOfPoint2f();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < minArea) continue;

            // Approximate polygon
            double peri = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approxCurve,
                    0.02 * peri, true);

            if (approxCurve.total() == 4) {
                Point[] corners = approxCurve.toArray();
                Point[] sorted = sortCorners(corners);

                // Smooth corners over time
                if (lastCorners != null && lastCorners.length == 4) {
                    double dist = 0;
                    for (int i = 0; i < 4; i++) {
                        dist += distance(sorted[i], lastCorners[i]);
                    }
                    double MIN_UPDATE_DIST = 35.0;
                    if (dist < MIN_UPDATE_DIST) {
                        sorted = smoothCorners(sorted, lastCorners);
                    }
                }

                lastCorners = sorted;
                return sorted;
            }
        }
        return null;
    }

    public Bitmap outputCheck(Bitmap bitmap){
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        // 1️⃣ Convert to grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);

        //Remove background by thresholding
        Mat mask = new Mat();
        Imgproc.adaptiveThreshold(
                grayMat,                           // input
                mask,                              // output
                255,                               // max value
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, // adaptive method
                Imgproc.THRESH_BINARY,             // threshold type
                11,                                // block size
                2                                  // C constant
        );
        // 3️⃣ Keep only the document (white areas in mask)
        Mat foreground = new Mat();
        mat.copyTo(foreground, mask);

        Imgproc.GaussianBlur(foreground, foreground, new Size(5, 5), 0);
        Mat dst = new Mat();

        // Define a sharpening kernel
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        float[] data = {
                0, -1,  0,
                -1,  5, -1,
                0, -1,  0
        };
        kernel.put(0, 0, data);

        // Apply the filter
        Imgproc.filter2D(foreground, dst, foreground.depth(), kernel);

        Imgproc.GaussianBlur(dst, dst, new Size(5, 5), 0);

        // 3️⃣ Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(dst, edges, 0, 150); // Adjust thresholds if needed

        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel2);

        Bitmap outputBitmap = Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(edges, outputBitmap);
        return outputBitmap;
    }

    /* Warp image using detected corners */
    public Bitmap warpToDocumentFromPoints(Bitmap bitmap, Point[] points) {
        Mat src = new Mat();
        Utils.bitmapToMat(bitmap, src);

        Point[] sorted = sortCorners(points);
        Point tl = sorted[0], tr = sorted[1], br = sorted[2], bl = sorted[3];

        double widthTop = Math.hypot(tr.x - tl.x, tr.y - tl.y);
        double widthBottom = Math.hypot(br.x - bl.x, br.y - bl.y);
        double maxWidth = Math.max(widthTop, widthBottom);

        double heightLeft = Math.hypot(bl.x - tl.x, bl.y - tl.y);
        double heightRight = Math.hypot(br.x - tr.x, br.y - tr.y);
        double maxHeight = Math.max(heightLeft, heightRight);

        if (maxWidth < 1) maxWidth = 1;
        if (maxHeight < 1) maxHeight = 1;

        Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
        srcPoints.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);

        Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);
        dstPoints.put(0, 0,
                0.0, 0.0,
                maxWidth - 1, 0.0,
                maxWidth - 1, maxHeight - 1,
                0.0, maxHeight - 1);

        Mat warpMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Mat dst = new Mat((int) maxHeight, (int) maxWidth, src.type());
        Imgproc.warpPerspective(src, dst, warpMat, dst.size());

        Bitmap output = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dst, output);
        return output;
    }

    // --- Utility helpers ---
    private Point[] sortCorners(Point[] pts) {
        Arrays.sort(pts, Comparator.comparingDouble(p -> p.y)); // top->bottom
        Point[] top = Arrays.copyOfRange(pts, 0, 2);
        Point[] bottom = Arrays.copyOfRange(pts, 2, 4);

        if (top[0].x > top[1].x) { Point temp = top[0]; top[0] = top[1]; top[1] = temp; }
        if (bottom[0].x > bottom[1].x) { Point temp = bottom[0]; bottom[0] = bottom[1]; bottom[1] = temp; }

        return new Point[]{top[0], top[1], bottom[1], bottom[0]};
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    private Point[] smoothCorners(Point[] newCorners, Point[] lastCorners) {
        Point[] smoothed = new Point[4];
        for (int i = 0; i < 4; i++) {
            double SMOOTH_ALPHA = 0.30;
            double x = SMOOTH_ALPHA * newCorners[i].x + (1 - SMOOTH_ALPHA) * lastCorners[i].x;
            double y = SMOOTH_ALPHA * newCorners[i].y + (1 - SMOOTH_ALPHA) * lastCorners[i].y;
            smoothed[i] = new Point(x, y);
        }
        return smoothed;
    }
}