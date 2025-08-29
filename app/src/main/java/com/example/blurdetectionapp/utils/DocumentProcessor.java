package com.example.blurdetectionapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class for document detection and corner processing
 */
public class DocumentProcessor {

    private static final String TAG = "DocumentProcessor";

    /**
     * Result class for document detection
     */
    public static class DocumentDetectionResult {
        public final Bitmap processedImage;
        public final boolean documentDetected;
        public final String statusMessage;
        public final Point[] corners;

        public DocumentDetectionResult(Bitmap processed, boolean detected, String status, Point[] corners) {
            this.processedImage = processed;
            this.documentDetected = detected;
            this.statusMessage = status;
            this.corners = corners;
        }
    }

    /**
     * Detect document corners and apply perspective correction
     * @param bitmap The input image
     * @return DocumentDetectionResult containing processed image and detection info
     */
    public static DocumentDetectionResult detectAndProcessDocument(Bitmap bitmap) {
        if (bitmap == null) {
            return new DocumentDetectionResult(null, false, "No image provided", null);
        }

        try {
            Mat src = new Mat();
            Utils.bitmapToMat(bitmap, src);

            Mat gray = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

            Mat edges = new Mat();
            Imgproc.Canny(gray, edges, 75, 200);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            double maxArea = -1;
            MatOfPoint2f bestContour = null;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > 1000) { // ignore small noise
                    MatOfPoint2f approxCurve = new MatOfPoint2f();
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                    Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * Imgproc.arcLength(contour2f, true), true);

                    if (approxCurve.total() == 4 && area > maxArea) {
                        maxArea = area;
                        bestContour = approxCurve;
                    }
                }
            }

            if (bestContour != null) {
                Point[] points = bestContour.toArray();
                Point[] sortedPoints = sortPoints(points);

                Bitmap processedBitmap = applyPerspectiveCorrection(src, sortedPoints);

                Log.d(TAG, "Document detected and processed successfully");
                return new DocumentDetectionResult(processedBitmap, true,
                        "Document detected and corrected", sortedPoints);

            } else {
                Log.d(TAG, "No document detected in image");
                return new DocumentDetectionResult(null, false,
                        "No document detected", null);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error processing document", e);
            return new DocumentDetectionResult(null, false,
                    "Processing failed: " + e.getMessage(), null);
        }
    }

    /**
     * Sort points to Top-Left, Top-Right, Bottom-Right, Bottom-Left order
     */
    private static Point[] sortPoints(Point[] points) {
        // Sort points based on their position
        Arrays.sort(points, (p1, p2) -> Double.compare(p1.y + p1.x, p2.y + p2.x));
        Point tl = points[0]; // Top-left has smallest sum
        Point br = points[3]; // Bottom-right has largest sum

        Arrays.sort(points, (p1, p2) -> Double.compare(p1.x - p1.y, p2.x - p2.y));
        Point tr = points[3]; // Top-right has largest diff (x-y)
        Point bl = points[0]; // Bottom-left has smallest diff (x-y)

        return new Point[]{tl, tr, br, bl};
    }

    /**
     * Apply perspective correction to transform document to rectangle
     */
    private static Bitmap applyPerspectiveCorrection(Mat src, Point[] corners) {
        Point tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];

        // Calculate dimensions of the corrected document
        double widthTop = Math.hypot(tr.x - tl.x, tr.y - tl.y);
        double widthBottom = Math.hypot(br.x - bl.x, br.y - bl.y);
        double maxWidth = Math.max(widthTop, widthBottom);

        double heightLeft = Math.hypot(bl.x - tl.x, bl.y - tl.y);
        double heightRight = Math.hypot(br.x - tr.x, br.y - tr.y);
        double maxHeight = Math.max(heightLeft, heightRight);

        // Define source and destination points for perspective transformation
        Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
        srcPoints.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);

        Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);
        dstPoints.put(0, 0,
                0.0, 0.0,
                maxWidth - 1, 0.0,
                maxWidth - 1, maxHeight - 1,
                0.0, maxHeight - 1);

        // Apply perspective transformation
        Mat warpMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Mat dst = new Mat((int) maxHeight, (int) maxWidth, CvType.CV_8UC4);
        Imgproc.warpPerspective(src, dst, warpMat, dst.size());

        // Convert back to bitmap
        Bitmap output = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dst, output);

        return output;
    }

    /**
     * Simple document detection method (backward compatibility)
     * @param bitmap The input image
     * @return Processed bitmap or null if no document detected
     */
    public static Bitmap detectDocumentCorners(Bitmap bitmap) {
        DocumentDetectionResult result = detectAndProcessDocument(bitmap);
        return result.processedImage;
    }
}