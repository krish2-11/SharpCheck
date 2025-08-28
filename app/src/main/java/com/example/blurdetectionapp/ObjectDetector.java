package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ObjectDetector {

    private static final String TAG = "ObjectDetector";

    // Enhanced parameters with smart filtering for small object detection
    private static final double MIN_CONTOUR_AREA_RATIO = 0.0002;  // Slightly less sensitive
    private static final double MAX_CONTOUR_AREA_RATIO = 0.85;
    private static final double CONTOUR_APPROXIMATION_EPSILON = 0.01;
    private static final int MIN_CONTOUR_POINTS = 3;

    // Color-based detection parameters - optimized for texture filtering
    private static final int GAUSSIAN_BLUR_SIZE = 2; // Slight blur to reduce texture noise
    private static final int BILATERAL_FILTER_D = 5;
    private static final double BILATERAL_SIGMA_COLOR = 50;
    private static final double BILATERAL_SIGMA_SPACE = 50;
    private static final int MORPH_KERNEL_SIZE = 2;

    // Smart filtering parameters
    private static final int MIN_OBJECT_SIZE = 8; // Slightly larger to avoid texture noise
    private static final double MIN_BRIGHTNESS_DIFFERENCE = 30; // Minimum brightness difference from background
    private static final double MIN_OBJECT_SOLIDITY = 0.3; // Minimum solidity to filter texture noise

    /**
     * Enhanced color-based object detection without grayscale conversion
     */
    public ObjectDetectionResult detectObjects(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            Log.e(TAG, "Invalid bitmap for object detection");
            return createEmptyResult(bitmap);
        }

        Log.d(TAG, "Starting enhanced color object detection on image: " + bitmap.getWidth() + "x" + bitmap.getHeight());

        List<DetectedObject> allDetectedObjects = new ArrayList<>();
        Mat originalSrc = null;
        Mat processed = null;

        try {
            // Convert bitmap to OpenCV Mat (keep color)
            originalSrc = new Mat();
            Utils.bitmapToMat(bitmap, originalSrc);
            processed = originalSrc.clone();

            // Method 1: Brightness contrast with smart background analysis
            List<DetectedObject> contrastObjects = detectWithSmartBrightnessContrast(originalSrc);

            // Method 2: HSV color segmentation with texture filtering
            List<DetectedObject> hsvObjects = detectWithHSVSegmentation(originalSrc);

            // Method 3: Shape-based filtering
            List<DetectedObject> shapeObjects = detectWithShapeAnalysis(originalSrc);

            // Combine results with smart filtering
            allDetectedObjects.addAll(contrastObjects);
            allDetectedObjects.addAll(hsvObjects);
            allDetectedObjects.addAll(shapeObjects);

            // Remove duplicates and keep best objects
            List<DetectedObject> finalObjects = filterAndDeduplicateObjects(allDetectedObjects);

            // Re-number objects
            for (int i = 0; i < finalObjects.size(); i++) {
                finalObjects.set(i, createRenumberedObject(finalObjects.get(i), i + 1));
            }

            // Draw contours on processed image
            drawContoursOnImage(processed, finalObjects);

            // Convert processed image back to bitmap
            Bitmap processedBitmap = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(processed, processedBitmap);

            ObjectDetectionResult result = new ObjectDetectionResult(bitmap, processedBitmap, finalObjects);

            Log.d(TAG, "Enhanced color object detection completed. Found " + finalObjects.size() + " objects");

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error in enhanced color object detection", e);
            return createEmptyResult(bitmap);
        } finally {
            if (originalSrc != null) originalSrc.release();
            if (processed != null) processed.release();
        }
    }

    /**
     * Method 1: HSV-based color segmentation
     */
    private List<DetectedObject> detectWithHSVSegmentation(Mat src) {
        Mat hsv = null;
        Mat mask = null;
        Mat combinedMask = null;
        Mat kernel = null;
        Mat hierarchy = null;

        try {
            Log.d(TAG, "Starting HSV segmentation detection");

            // Convert BGR to HSV
            hsv = new Mat();
            Imgproc.cvtColor(src, hsv, Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV);

            combinedMask = Mat.zeros(hsv.rows(), hsv.cols(), CvType.CV_8UC1);

            // Define multiple color ranges for common object colors
            List<Scalar[]> colorRanges = new ArrayList<>();

            // Brown/Wood colors (furniture)
            colorRanges.add(new Scalar[]{new Scalar(8, 50, 20), new Scalar(20, 255, 200)});

            // Blue colors (common in furniture, containers)
            colorRanges.add(new Scalar[]{new Scalar(100, 50, 50), new Scalar(130, 255, 255)});

            // Red colors
            colorRanges.add(new Scalar[]{new Scalar(0, 50, 50), new Scalar(10, 255, 255)});
            colorRanges.add(new Scalar[]{new Scalar(170, 50, 50), new Scalar(180, 255, 255)});

            // Green colors
            colorRanges.add(new Scalar[]{new Scalar(40, 50, 50), new Scalar(80, 255, 255)});

            // Yellow colors
            colorRanges.add(new Scalar[]{new Scalar(20, 50, 50), new Scalar(35, 255, 255)});

            // Purple/Violet colors
            colorRanges.add(new Scalar[]{new Scalar(130, 50, 50), new Scalar(170, 255, 255)});

            // White/Light colors (with some saturation to avoid pure background)
            colorRanges.add(new Scalar[]{new Scalar(0, 0, 200), new Scalar(180, 30, 255)});

            // Dark colors (black furniture, etc.)
            colorRanges.add(new Scalar[]{new Scalar(0, 0, 0), new Scalar(180, 255, 50)});

            // Create masks for each color range and combine
            for (Scalar[] range : colorRanges) {
                mask = new Mat();
                Core.inRange(hsv, range[0], range[1], mask);
                Core.bitwise_or(combinedMask, mask, combinedMask);
                mask.release();
            }

            // Morphological operations to clean up
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
            Imgproc.morphologyEx(combinedMask, combinedMask, Imgproc.MORPH_OPEN, kernel);
            Imgproc.morphologyEx(combinedMask, combinedMask, Imgproc.MORPH_CLOSE, kernel);

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            hierarchy = new Mat();
            Imgproc.findContours(combinedMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            Log.d(TAG, "HSV segmentation found " + contours.size() + " contours");

            return analyzeContours(contours, src, "HSV");

        } catch (Exception e) {
            Log.e(TAG, "Error in HSV segmentation", e);
            return new ArrayList<>();
        } finally {
            if (hsv != null) hsv.release();
            if (mask != null) mask.release();
            if (combinedMask != null) combinedMask.release();
            if (kernel != null) kernel.release();
            if (hierarchy != null) hierarchy.release();
        }
    }

    /**
     * Method 1: Smart brightness contrast with background analysis
     */
    private List<DetectedObject> detectWithSmartBrightnessContrast(Mat src) {
        Mat gray = null;
        Mat binary = null;
        Mat kernel = null;
        Mat hierarchy = null;

        try {
            Log.d(TAG, "Starting smart brightness contrast detection");

            // Convert to grayscale
            gray = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);

            // Calculate background brightness (median value)
            Mat hist = new Mat();
            List<Mat> images = new ArrayList<>();
            images.add(gray);
            Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, new MatOfInt(256), new MatOfFloat(0, 256));

            // Find median brightness value as background reference
            float[] histData = new float[(int) hist.total()];
            hist.get(0, 0, histData);

            int totalPixels = gray.rows() * gray.cols();
            int medianPos = totalPixels / 2;
            int currentSum = 0;
            int backgroundBrightness = 128; // Default fallback

            for (int i = 0; i < histData.length; i++) {
                currentSum += histData[i];
                if (currentSum >= medianPos) {
                    backgroundBrightness = i;
                    break;
                }
            }
            hist.release();

            Log.d(TAG, "Detected background brightness: " + backgroundBrightness);

            // Apply gentle blur to reduce texture noise while preserving objects
            Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0.5);

            // Use adaptive threshold with background-aware parameters
            binary = new Mat();
            int thresholdValue = Math.max(backgroundBrightness + 25, 180); // At least 25 points brighter than background
            Imgproc.threshold(gray, binary, thresholdValue, 255, Imgproc.THRESH_BINARY);

            // Combine with adaptive threshold for local variations
            Mat adaptiveBinary = new Mat();
            Imgproc.adaptiveThreshold(gray, adaptiveBinary, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 8);

            // Take intersection of both methods (objects must pass both tests)
            Core.bitwise_and(binary, adaptiveBinary, binary);
            adaptiveBinary.release();

            // Minimal morphological operations
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2, 2));
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 1);

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            Log.d(TAG, "Smart brightness contrast found " + contours.size() + " contours");

            // Filter contours with brightness analysis
            return analyzeContoursWithBrightness(contours, src, gray, backgroundBrightness, "SmartBrightness");

        } catch (Exception e) {
            Log.e(TAG, "Error in smart brightness contrast detection", e);
            return new ArrayList<>();
        } finally {
            if (gray != null) gray.release();
            if (binary != null) binary.release();
            if (kernel != null) kernel.release();
            if (hierarchy != null) hierarchy.release();
        }
    }

    /**
     * Method 3: Shape-based analysis for rice grain characteristics
     */
    private List<DetectedObject> detectWithShapeAnalysis(Mat src) {
        Mat gray = null;
        Mat binary = null;
        Mat kernel = null;
        Mat hierarchy = null;

        try {
            Log.d(TAG, "Starting shape-based analysis");

            // Convert to grayscale
            gray = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);

            // Apply bilateral filter to smooth texture while preserving edges
            Mat filtered = new Mat();
            Imgproc.bilateralFilter(gray, filtered, 9, 75, 75);

            // Multiple threshold approaches
            binary = new Mat();
            Mat binary1 = new Mat();
            Mat binary2 = new Mat();

            // High threshold for very bright objects (rice grains)
            Imgproc.threshold(filtered, binary1, 200, 255, Imgproc.THRESH_BINARY);

            // OTSU threshold
            Imgproc.threshold(filtered, binary2, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

            // Take intersection for more selective detection
            Core.bitwise_and(binary1, binary2, binary);
            binary1.release();
            binary2.release();
            filtered.release();

            // Shape-preserving morphological operations
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2, 2));
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel);

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            Log.d(TAG, "Shape analysis found " + contours.size() + " contours");

            // Analyze contours with shape filtering
            return analyzeContoursWithShapeFilter(contours, src, "Shape");

        } catch (Exception e) {
            Log.e(TAG, "Error in shape analysis", e);
            return new ArrayList<>();
        } finally {
            if (gray != null) gray.release();
            if (binary != null) binary.release();
            if (kernel != null) kernel.release();
            if (hierarchy != null) hierarchy.release();
        }
    }

    /**
     * Analyze contours with brightness filtering to avoid background texture
     */
    private List<DetectedObject> analyzeContoursWithBrightness(List<MatOfPoint> contours, Mat srcImage, Mat grayImage, int backgroundBrightness, String method) {
        List<DetectedObject> detectedObjects = new ArrayList<>();

        if (contours.isEmpty()) {
            return detectedObjects;
        }

        int imageArea = srcImage.rows() * srcImage.cols();
        double minArea = MIN_CONTOUR_AREA_RATIO * imageArea;
        double maxArea = MAX_CONTOUR_AREA_RATIO * imageArea;

        for (int i = 0; i < contours.size(); i++) {
            try {
                MatOfPoint contour = contours.get(i);
                if (contour == null || contour.empty()) continue;

                double area = Imgproc.contourArea(contour);
                if (area < minArea || area > maxArea) continue;

                org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);
                if (boundingRect.width < MIN_OBJECT_SIZE || boundingRect.height < MIN_OBJECT_SIZE) continue;

                // Calculate average brightness within the contour
                Mat mask = Mat.zeros(grayImage.rows(), grayImage.cols(), CvType.CV_8UC1);
                List<MatOfPoint> contourList = new ArrayList<>();
                contourList.add(contour);
                Imgproc.drawContours(mask, contourList, -1, new Scalar(255), -1);

                Scalar meanBrightness = Core.mean(grayImage, mask);
                mask.release();

                double objectBrightness = meanBrightness.val[0];
                double brightnessDifference = objectBrightness - backgroundBrightness;

                // Filter objects that aren't significantly brighter than background
                if (brightnessDifference < MIN_BRIGHTNESS_DIFFERENCE) {
                    Log.d(TAG, method + " - Filtered object " + i + " due to low brightness difference: " + brightnessDifference);
                    continue;
                }

                // Additional shape validation for rice grains
                if (!isRiceGrainShape(contour, area)) {
                    Log.d(TAG, method + " - Filtered object " + i + " due to non-rice shape");
                    continue;
                }

                DetectedObject obj = analyzeContour(contour, detectedObjects.size() + 1);
                if (obj != null && isValidSmartObject(obj)) {
                    detectedObjects.add(obj);
                    Log.d(TAG, method + " - Valid object " + obj.getId() + ": area=" + area + ", brightness=" + (int)objectBrightness);
                }

            } catch (Exception e) {
                Log.w(TAG, "Error processing contour " + i + " in " + method, e);
            }
        }

        Log.d(TAG, method + " - Final detected objects after brightness filtering: " + detectedObjects.size());
        return detectedObjects;
    }

    /**
     * Analyze contours with shape filtering
     */
    private List<DetectedObject> analyzeContoursWithShapeFilter(List<MatOfPoint> contours, Mat srcImage, String method) {
        List<DetectedObject> detectedObjects = new ArrayList<>();

        if (contours.isEmpty()) {
            return detectedObjects;
        }

        int imageArea = srcImage.rows() * srcImage.cols();
        double minArea = MIN_CONTOUR_AREA_RATIO * imageArea;
        double maxArea = MAX_CONTOUR_AREA_RATIO * imageArea;

        for (int i = 0; i < contours.size(); i++) {
            try {
                MatOfPoint contour = contours.get(i);
                if (contour == null || contour.empty()) continue;

                double area = Imgproc.contourArea(contour);
                if (area < minArea || area > maxArea) continue;

                org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);
                if (boundingRect.width < MIN_OBJECT_SIZE || boundingRect.height < MIN_OBJECT_SIZE) continue;

                // Enhanced rice grain shape validation
                if (!isRiceGrainShape(contour, area)) {
                    continue;
                }

                DetectedObject obj = analyzeContour(contour, detectedObjects.size() + 1);
                if (obj != null && isValidSmartObject(obj)) {
                    detectedObjects.add(obj);
                    Log.d(TAG, method + " - Valid object " + obj.getId() + ": area=" + area);
                }

            } catch (Exception e) {
                Log.w(TAG, "Error processing contour " + i + " in " + method, e);
            }
        }

        Log.d(TAG, method + " - Final detected objects after shape filtering: " + detectedObjects.size());
        return detectedObjects;
    }

    /**
     * Check if contour has rice grain-like characteristics
     */
    private boolean isRiceGrainShape(MatOfPoint contour, double area) {
        try {
            // Calculate contour properties
            double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            if (perimeter <= 0) return false;

            // Solidity (area / convex hull area) - rice grains should be fairly solid
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);

            int[] hullIndices = hull.toArray();
            Point[] contourPoints = contour.toArray();
            Point[] hullPoints = new Point[hullIndices.length];
            for (int i = 0; i < hullIndices.length; i++) {
                hullPoints[i] = contourPoints[hullIndices[i]];
            }

            MatOfPoint hullContour = new MatOfPoint();
            hullContour.fromArray(hullPoints);
            double hullArea = Imgproc.contourArea(hullContour);

            hull.release();
            hullContour.release();

            double solidity = hullArea > 0 ? area / hullArea : 0;

            // Roundness (4*pi*area / perimeter^2)
            double roundness = (4 * Math.PI * area) / (perimeter * perimeter);

            // Aspect ratio
            org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);
            double aspectRatio = Math.max(boundingRect.width, boundingRect.height) /
                    (double) Math.min(boundingRect.width, boundingRect.height);

            // Rice grain characteristics:
            // - Solidity > 0.3 (fairly solid shape, not too irregular)
            // - Roundness between 0.1 and 0.9 (oval-ish, not too circular or too thin)
            // - Aspect ratio < 10 (not extremely elongated)
            // - Area between reasonable bounds for rice grains

            boolean validSolidity = solidity > MIN_OBJECT_SOLIDITY;
            boolean validRoundness = roundness > 0.1 && roundness < 0.9;
            boolean validAspectRatio = aspectRatio < 10;
            boolean validArea = area > 15 && area < 2000; // Reasonable rice grain size range

            return validSolidity && validRoundness && validAspectRatio && validArea;

        } catch (Exception e) {
            Log.w(TAG, "Error in rice grain shape validation", e);
            return false;
        }
    }

    /**
     * Analyze contours and create detected objects
     */
    private List<DetectedObject> analyzeContours(List<MatOfPoint> contours, Mat srcImage, String method) {
        List<DetectedObject> detectedObjects = new ArrayList<>();

        if (contours.isEmpty()) {
            Log.d(TAG, method + " - No contours to analyze");
            return detectedObjects;
        }

        int imageArea = srcImage.rows() * srcImage.cols();
        double minArea = MIN_CONTOUR_AREA_RATIO * imageArea;
        double maxArea = MAX_CONTOUR_AREA_RATIO * imageArea;

        Log.d(TAG, method + " - Analyzing " + contours.size() + " contours with area range: " + minArea + " - " + maxArea);

        for (int i = 0; i < contours.size(); i++) {
            try {
                MatOfPoint contour = contours.get(i);

                if (contour == null || contour.empty()) {
                    continue;
                }

                double area = Imgproc.contourArea(contour);

                // Area filtering
                if (area < minArea || area > maxArea) {
                    continue;
                }

                // Check minimum bounding box size
                org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);
                if (boundingRect.width < MIN_OBJECT_SIZE || boundingRect.height < MIN_OBJECT_SIZE) {
                    continue;
                }

                // Create detected object
                DetectedObject obj = analyzeContour(contour, detectedObjects.size() + 1);
                if (obj != null ) {
                    detectedObjects.add(obj);
                    Log.d(TAG, method + " - Valid object " + obj.getId() + ": area=" + area + ", shape=" + obj.getShapeType());
                }

            } catch (Exception e) {
                Log.w(TAG, "Error processing contour " + i + " in " + method, e);
            }
        }

        Log.d(TAG, method + " - Final detected objects: " + detectedObjects.size());
        return detectedObjects;
    }

    /**
     * Analyze individual contour
     */
    private DetectedObject analyzeContour(MatOfPoint contour, int objectId) {
        MatOfPoint2f contour2f = null;
        MatOfPoint2f approx = null;

        try {
            double area = Imgproc.contourArea(contour);
            double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);

            if (area <= 0 || perimeter <= 0) {
                return null;
            }

            org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);

            // Approximate contour to polygon
            contour2f = new MatOfPoint2f(contour.toArray());
            approx = new MatOfPoint2f();
            double epsilon = CONTOUR_APPROXIMATION_EPSILON * perimeter;
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

            double aspectRatio = Math.max(boundingRect.width, boundingRect.height) /
                    (double) Math.min(boundingRect.width, boundingRect.height);

            double roundness = (4 * Math.PI * area) / (perimeter * perimeter);

            String shapeType = determineShapeType((int) approx.total(), roundness, aspectRatio, area);

            Point center = new Point(boundingRect.x + boundingRect.width / 2.0,
                    boundingRect.y + boundingRect.height / 2.0);

            // Create contour copy
            MatOfPoint contourCopy = new MatOfPoint();
            Point[] originalPoints = contour.toArray();
            contourCopy.fromArray(originalPoints);

            return new DetectedObject(objectId, area, perimeter, roundness, aspectRatio,
                    center, boundingRect, shapeType, contourCopy);

        } catch (Exception e) {
            Log.w(TAG, "Error analyzing contour for object " + objectId, e);
            return null;
        } finally {
            if (contour2f != null) contour2f.release();
            if (approx != null) approx.release();
        }
    }

    /**
     * Enhanced shape determination with size-based categorization
     */
    private String determineShapeType(int vertices, double roundness, double aspectRatio, double area) {
        // Size-based initial categorization
        String sizeCategory;
        if (area < 1000) {
            sizeCategory = "Small";
        } else if (area < 10000) {
            sizeCategory = "Medium";
        } else {
            sizeCategory = "Large";
        }

        // Shape analysis
        String shapeType;
        if (vertices == 4) {
            if (aspectRatio < 1.3) {
                shapeType = sizeCategory + " Square Object";
            } else if (aspectRatio < 3.0) {
                shapeType = sizeCategory + " Rectangular Object";
            } else {
                shapeType = sizeCategory + " Long Object";
            }
        } else if (roundness > 0.7) {
            if (aspectRatio < 1.2) {
                shapeType = sizeCategory + " Round Object";
            } else {
                shapeType = sizeCategory + " Oval Object";
            }
        } else if (vertices == 3) {
            shapeType = sizeCategory + " Triangular Object";
        } else if (vertices >= 5 && vertices <= 8) {
            shapeType = sizeCategory + " Polygon Object";
        } else if (vertices > 8) {
            if (roundness > 0.5) {
                shapeType = sizeCategory + " Curved Object";
            } else {
                shapeType = sizeCategory + " Complex Object";
            }
        } else {
            shapeType = sizeCategory + " Irregular Object";
        }

        return shapeType;
    }

    /**
     * Enhanced object validation with smart filtering
     */
    private boolean isValidSmartObject(DetectedObject obj) {
        // Area validation - more restrictive to avoid texture noise
        if (obj.getArea() < 15 || obj.getArea() > 2000) {
            return false;
        }

        // Aspect ratio - rice grains shouldn't be extremely elongated
        if (obj.getAspectRatio() > 10) {
            return false;
        }

        // Roundness - should be somewhat oval/rounded
        if (obj.getRoundness() < 0.1 || obj.getRoundness() > 0.9) {
            return false;
        }

        // Bounding box validation - avoid very thin objects
        org.opencv.core.Rect rect = obj.getBoundingRect();
        if (rect.width < 3 || rect.height < 3) {
            return false;
        }

        return true;
    }

    /**
     * Filter and deduplicate overlapping objects - more lenient for small objects
     */
    private List<DetectedObject> filterAndDeduplicateObjects(List<DetectedObject> objects) {
        if (objects.isEmpty()) {
            return objects;
        }

        // Sort by area (largest first)
        Collections.sort(objects, new Comparator<DetectedObject>() {
            @Override
            public int compare(DetectedObject o1, DetectedObject o2) {
                return Double.compare(o2.getArea(), o1.getArea());
            }
        });

        List<DetectedObject> filtered = new ArrayList<>();

        for (DetectedObject obj : objects) {
            boolean isDuplicate = false;

            for (DetectedObject existing : filtered) {
                // Use different overlap thresholds based on object size
                double overlapThreshold = obj.getArea() < 100 ? 0.1 : 0.3; // 10% for small objects, 30% for larger

                if (isOverlapping(obj, existing, overlapThreshold)) {
                    isDuplicate = true;
                    break;
                }
            }

            if (!isDuplicate) {
                filtered.add(obj);
            }
        }

        Log.d(TAG, "Deduplicated objects: " + objects.size() + " -> " + filtered.size());
        return filtered;
    }

    /**
     * Check if two objects are overlapping
     */
    private boolean isOverlapping(DetectedObject obj1, DetectedObject obj2, double threshold) {
        org.opencv.core.Rect rect1 = obj1.getBoundingRect();
        org.opencv.core.Rect rect2 = obj2.getBoundingRect();

        // Calculate intersection area
        int x1 = Math.max(rect1.x, rect2.x);
        int y1 = Math.max(rect1.y, rect2.y);
        int x2 = Math.min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = Math.min(rect1.y + rect1.height, rect2.y + rect2.height);

        if (x1 >= x2 || y1 >= y2) {
            return false;
        }

        double intersectionArea = (x2 - x1) * (y2 - y1);
        double minArea = Math.min(rect1.area(), rect2.area());

        return (intersectionArea / minArea) > threshold;
    }

    /**
     * Create a renumbered copy of detected object
     */
    private DetectedObject createRenumberedObject(DetectedObject original, int newId) {
        return new DetectedObject(
                newId,
                original.getArea(),
                original.getPerimeter(),
                original.getRoundness(),
                original.getAspectRatio(),
                original.getCenter(),
                original.getBoundingRect(),
                original.getShapeType(),
                original.getContour()
        );
    }

    /**
     * Enhanced contour drawing with better colors
     */
    private void drawContoursOnImage(Mat image, List<DetectedObject> objects) {
        if (objects.isEmpty()) {
            return;
        }

        Scalar[] colors = {
                new Scalar(0, 255, 0, 255),    // Bright Green
                new Scalar(255, 0, 0, 255),    // Red
                new Scalar(0, 0, 255, 255),    // Blue
                new Scalar(255, 255, 0, 255),  // Yellow
                new Scalar(255, 0, 255, 255),  // Magenta
                new Scalar(0, 255, 255, 255),  // Cyan
                new Scalar(255, 165, 0, 255),  // Orange
                new Scalar(128, 0, 128, 255),  // Purple
                new Scalar(255, 192, 203, 255), // Pink
                new Scalar(0, 128, 0, 255)     // Dark Green
        };

        for (DetectedObject obj : objects) {
            try {
                int colorIndex = (obj.getId() - 1) % colors.length;
                Scalar color = colors[colorIndex];

                List<MatOfPoint> contours = new ArrayList<>();
                contours.add(obj.getContour());

                // Draw thick contour outline
                Imgproc.drawContours(image, contours, -1, color, 3);

                // Draw center point
                Imgproc.circle(image, obj.getCenter(), 6, new Scalar(255, 255, 255, 255), -1);
                Imgproc.circle(image, obj.getCenter(), 6, color, 2);

                // Draw object ID
                String label = String.valueOf(obj.getId());
                Point textPos = new Point(obj.getBoundingRect().x, obj.getBoundingRect().y - 10);

                // Text background
                Imgproc.rectangle(image,
                        new Point(textPos.x - 3, textPos.y - 20),
                        new Point(textPos.x + 25, textPos.y + 5),
                        new Scalar(0, 0, 0, 180), -1);

                // Text
                Imgproc.putText(image, label, textPos, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
                        new Scalar(255, 255, 255, 255), 2);

            } catch (Exception e) {
                Log.w(TAG, "Error drawing object " + obj.getId(), e);
            }
        }
    }

    private ObjectDetectionResult createEmptyResult(Bitmap originalBitmap) {
        return new ObjectDetectionResult(originalBitmap, originalBitmap, new ArrayList<>());
    }
}