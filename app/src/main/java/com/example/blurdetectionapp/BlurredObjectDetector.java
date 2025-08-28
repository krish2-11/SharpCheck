package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

/**
 * Detects blurred objects in images using multiple detection methods
 * Focuses on small objects like rice, beans, etc.
 */
public class BlurredObjectDetector {
    private static final String TAG = "BlurredObjectDetector";

    // Blur detection thresholds
    private static final double LAPLACIAN_THRESHOLD = 100.0;
    private static final double SOBEL_THRESHOLD = 50.0;
    private static final double VARIANCE_THRESHOLD = 30.0;

    // Object detection parameters
    private static final int MIN_OBJECT_AREA = 50;
    private static final int MAX_OBJECT_AREA = 5000;
    private static final double MIN_CIRCULARITY = 0.3;
    private static final double MAX_ASPECT_RATIO = 3.0;

    public BlurredObjectDetectionResult detectBlurredObjects(Bitmap inputBitmap) {
        Log.d(TAG, "Starting blurred object detection");

        try {
            // Convert bitmap to OpenCV Mat
            Mat inputMat = new Mat();
            Utils.bitmapToMat(inputBitmap, inputMat);

            // Convert to grayscale
            Mat grayMat = new Mat();
            Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_RGB2GRAY);

            // Detect potential objects first
            List<BlurredObject> detectedObjects = detectPotentialObjects(grayMat, inputMat);

            // Analyze blur for each detected object
            List<BlurredObject> blurredObjects = new ArrayList<>();
            for (BlurredObject obj : detectedObjects) {
                if (analyzeObjectBlur(grayMat, obj)) {
                    blurredObjects.add(obj);
                    Log.d(TAG, "Found blurred object: " + obj.toString());
                }
            }

            Log.d(TAG, "Detection complete. Found " + blurredObjects.size() + " blurred objects out of " + detectedObjects.size() + " total objects");

            return new BlurredObjectDetectionResult(blurredObjects, detectedObjects.size());

        } catch (Exception e) {
            Log.e(TAG, "Error in blurred object detection", e);
            return new BlurredObjectDetectionResult(new ArrayList<>(), 0);
        }
    }

    private List<BlurredObject> detectPotentialObjects(Mat grayMat, Mat colorMat) {
        List<BlurredObject> objects = new ArrayList<>();

        try {
            // Method 1: Contour-based detection
            objects.addAll(detectByContours(grayMat, colorMat));

            // Method 2: Blob detection for circular objects
            objects.addAll(detectByBlobs(grayMat, colorMat));

            // Method 3: Watershed segmentation for touching objects
            objects.addAll(detectByWatershed(grayMat, colorMat));

            // Remove duplicates and merge overlapping detections
            objects = mergeDuplicateDetections(objects);

        } catch (Exception e) {
            Log.e(TAG, "Error in object detection", e);
        }

        return objects;
    }

    private List<BlurredObject> detectByContours(Mat grayMat, Mat colorMat) {
        List<BlurredObject> objects = new ArrayList<>();

        try {
            // Preprocessing
            Mat processed = preprocessForContours(grayMat);

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(processed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Analyze each contour
            for (int i = 0; i < contours.size(); i++) {
                MatOfPoint contour = contours.get(i);

                // Calculate contour properties
                double area = Imgproc.contourArea(contour);
                if (area < MIN_OBJECT_AREA || area > MAX_OBJECT_AREA) {
                    continue;
                }

                // Get bounding rectangle
                Rect boundingRect = Imgproc.boundingRect(contour);

                // Calculate shape properties
                double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
                double circularity = 4 * Math.PI * area / (perimeter * perimeter);
                double aspectRatio = (double) boundingRect.width / boundingRect.height;

                // Filter based on shape characteristics
                if (circularity >= MIN_CIRCULARITY && aspectRatio <= MAX_ASPECT_RATIO) {
                    // Extract ROI
                    Mat roi = new Mat(grayMat, boundingRect);
                    Mat colorRoi = new Mat(colorMat, boundingRect);

                    BlurredObject obj = new BlurredObject(
                            boundingRect,
                            roi.clone(),
                            colorRoi.clone(),
                            area,
                            perimeter,
                            circularity,
                            aspectRatio,
                            "CONTOUR"
                    );

                    objects.add(obj);
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in contour detection", e);
        }

        return objects;
    }

    private List<BlurredObject> detectByBlobs(Mat grayMat, Mat colorMat) {
        List<BlurredObject> objects = new ArrayList<>();

        try {
            // Apply circular Hough Transform for blob-like objects
            Mat circles = new Mat();
            Imgproc.HoughCircles(grayMat, circles, Imgproc.HOUGH_GRADIENT, 1.0, 20,
                    50, 30, 5, 50); // Parameters tuned for small objects

            for (int i = 0; i < circles.cols(); i++) {
                double[] circle = circles.get(0, i);
                if (circle != null && circle.length >= 3) {
                    Point center = new Point(Math.round(circle[0]), Math.round(circle[1]));
                    int radius = (int) Math.round(circle[2]);

                    // Create bounding rectangle
                    Rect boundingRect = new Rect(
                            (int) Math.max(0, center.x - radius),
                            (int) Math.max(0, center.y - radius),
                            Math.min(2 * radius, grayMat.cols() - (int) (center.x - radius)),
                            Math.min(2 * radius, grayMat.rows() - (int) (center.y - radius))
                    );

                    if (boundingRect.width > 10 && boundingRect.height > 10) {
                        Mat roi = new Mat(grayMat, boundingRect);
                        Mat colorRoi = new Mat(colorMat, boundingRect);

                        double area = Math.PI * radius * radius;
                        double perimeter = 2 * Math.PI * radius;

                        BlurredObject obj = new BlurredObject(
                                boundingRect,
                                roi.clone(),
                                colorRoi.clone(),
                                area,
                                perimeter,
                                1.0, // Perfect circularity for detected circles
                                1.0, // Perfect aspect ratio for circles
                                "BLOB"
                        );

                        objects.add(obj);
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in blob detection", e);
        }

        return objects;
    }

    private List<BlurredObject> detectByWatershed(Mat grayMat, Mat colorMat) {
        List<BlurredObject> objects = new ArrayList<>();

        try {
            // Preprocessing for watershed
            Mat processed = new Mat();
            Imgproc.GaussianBlur(grayMat, processed, new Size(3, 3), 0);

            // Apply threshold
            Mat binary = new Mat();
            Imgproc.threshold(processed, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // Distance transform
            Mat dist = new Mat();
            Imgproc.distanceTransform(binary, dist, Imgproc.DIST_L2, 3);

            // Find local maxima as markers
            Mat markers = new Mat();
            Imgproc.threshold(dist, markers, 0.5 * 0.8, 255, Imgproc.THRESH_BINARY); // Reduced threshold for small objects

            markers.convertTo(markers, CvType.CV_8UC1);

            // Find contours of markers
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(markers, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Convert to watershed markers
            Mat markersWS = Mat.zeros(markers.size(), CvType.CV_32SC1);
            for (int i = 0; i < contours.size(); i++) {
                Imgproc.drawContours(markersWS, contours, i, new Scalar(i + 1), -1);
            }

            // Apply watershed
            Mat colorMat3Ch = new Mat();
            Imgproc.cvtColor(grayMat, colorMat3Ch, Imgproc.COLOR_GRAY2BGR);
            Imgproc.watershed(colorMat3Ch, markersWS);

            // Extract segmented regions
            for (int i = 1; i <= contours.size(); i++) {
                Mat mask = new Mat();
                Core.inRange(markersWS, new Scalar(i), new Scalar(i), mask);

                List<MatOfPoint> segmentContours = new ArrayList<>();
                Mat segmentHierarchy = new Mat();
                Imgproc.findContours(mask, segmentContours, segmentHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

                for (MatOfPoint contour : segmentContours) {
                    double area = Imgproc.contourArea(contour);
                    if (area >= MIN_OBJECT_AREA && area <= MAX_OBJECT_AREA) {
                        Rect boundingRect = Imgproc.boundingRect(contour);

                        Mat roi = new Mat(grayMat, boundingRect);
                        Mat colorRoi = new Mat(colorMat, boundingRect);

                        double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
                        double circularity = 4 * Math.PI * area / (perimeter * perimeter);
                        double aspectRatio = (double) boundingRect.width / boundingRect.height;

                        BlurredObject obj = new BlurredObject(
                                boundingRect,
                                roi.clone(),
                                colorRoi.clone(),
                                area,
                                perimeter,
                                circularity,
                                aspectRatio,
                                "WATERSHED"
                        );

                        objects.add(obj);
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in watershed detection", e);
        }

        return objects;
    }

    private Mat preprocessForContours(Mat grayMat) {
        Mat processed = new Mat();

        // Apply bilateral filter to reduce noise while preserving edges
        Imgproc.bilateralFilter(grayMat, processed, 9, 75, 75);

        // Apply morphological operations
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.morphologyEx(processed, processed, Imgproc.MORPH_CLOSE, kernel);

        // Apply adaptive threshold
        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(processed, binary, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 11, 2);

        return binary;
    }

    private boolean analyzeObjectBlur(Mat grayMat, BlurredObject obj) {
        try {
            Mat roi = obj.getGrayRoi();

            // Method 1: Laplacian Variance
            double laplacianVar = calculateLaplacianVariance(roi);

            // Method 2: Sobel Gradient
            double sobelMagnitude = calculateSobelMagnitude(roi);

            // Method 3: Variance of pixel intensities
            double pixelVariance = calculatePixelVariance(roi);

            // Method 4: Tenengrad focus measure
            double tenengrad = calculateTenengrad(roi);

            // Store blur metrics in object
            obj.setBlurMetrics(laplacianVar, sobelMagnitude, pixelVariance, tenengrad);

            // Determine if object is blurred based on multiple criteria
            boolean isBlurred = (laplacianVar < LAPLACIAN_THRESHOLD) ||
                    (sobelMagnitude < SOBEL_THRESHOLD) ||
                    (pixelVariance < VARIANCE_THRESHOLD);

            obj.setBlurred(isBlurred);

            Log.d(TAG, String.format("Object blur analysis - Laplacian: %.2f, Sobel: %.2f, Variance: %.2f, Tenengrad: %.2f, Blurred: %b",
                    laplacianVar, sobelMagnitude, pixelVariance, tenengrad, isBlurred));

            return isBlurred;

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing object blur", e);
            return false;
        }
    }

    private double calculateLaplacianVariance(Mat roi) {
        Mat laplacian = new Mat();
        Imgproc.Laplacian(roi, laplacian, CvType.CV_64F);

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(laplacian, mean, stddev);

        double variance = Math.pow(stddev.get(0, 0)[0], 2);
        return variance;
    }

    private double calculateSobelMagnitude(Mat roi) {
        Mat sobelX = new Mat();
        Mat sobelY = new Mat();
        Mat magnitude = new Mat();

        Imgproc.Sobel(roi, sobelX, CvType.CV_64F, 1, 0, 3);
        Imgproc.Sobel(roi, sobelY, CvType.CV_64F, 0, 1, 3);

        Core.magnitude(sobelX, sobelY, magnitude);

        Scalar meanScalar = Core.mean(magnitude);
        return meanScalar.val[0];
    }

    private double calculatePixelVariance(Mat roi) {
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(roi, mean, stddev);

        return Math.pow(stddev.get(0, 0)[0], 2);
    }

    private double calculateTenengrad(Mat roi) {
        Mat sobelX = new Mat();
        Mat sobelY = new Mat();

        Imgproc.Sobel(roi, sobelX, CvType.CV_64F, 1, 0, 3);
        Imgproc.Sobel(roi, sobelY, CvType.CV_64F, 0, 1, 3);

        // Square the gradients
        Core.multiply(sobelX, sobelX, sobelX);
        Core.multiply(sobelY, sobelY, sobelY);

        // Add squared gradients
        Mat gradSquared = new Mat();
        Core.add(sobelX, sobelY, gradSquared);

        // Sum all values
        Scalar sum = Core.sumElems(gradSquared);
        return sum.val[0];
    }

    private List<BlurredObject> mergeDuplicateDetections(List<BlurredObject> objects) {
        List<BlurredObject> merged = new ArrayList<>();
        boolean[] used = new boolean[objects.size()];

        for (int i = 0; i < objects.size(); i++) {
            if (used[i]) continue;

            BlurredObject current = objects.get(i);
            List<BlurredObject> overlapping = new ArrayList<>();
            overlapping.add(current);
            used[i] = true;

            // Find overlapping objects
            for (int j = i + 1; j < objects.size(); j++) {
                if (used[j]) continue;

                BlurredObject other = objects.get(j);
                if (calculateOverlap(current.getBoundingRect(), other.getBoundingRect()) > 0.5) {
                    overlapping.add(other);
                    used[j] = true;
                }
            }

            // Merge overlapping objects (keep the one with best detection confidence)
            BlurredObject best = overlapping.get(0);
            for (BlurredObject obj : overlapping) {
                if (obj.getArea() > best.getArea()) { // Use area as confidence metric
                    best = obj;
                }
            }

            merged.add(best);
        }

        return merged;
    }

    private double calculateOverlap(Rect rect1, Rect rect2) {
        int x1 = Math.max(rect1.x, rect2.x);
        int y1 = Math.max(rect1.y, rect2.y);
        int x2 = Math.min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = Math.min(rect1.y + rect1.height, rect2.y + rect2.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0;
        }

        int intersectionArea = (x2 - x1) * (y2 - y1);
        int unionArea = (int) (rect1.area() + rect2.area() - intersectionArea);

        return (double) intersectionArea / unionArea;
    }
}
