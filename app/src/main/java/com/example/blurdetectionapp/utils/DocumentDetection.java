package com.example.blurdetectionapp.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class DocumentDetection {

    private static final String TAG = "DocumentDetection";

    // --- FINAL TUNED PARAMETERS ---
    private static final double SMOOTHING_ALPHA = 0.25;
    private static final double JITTER_REJECTION_THRESHOLD_PERCENT = 0.03; // 3% of image width
    private static final double MOVEMENT_RESET_THRESHOLD_PERCENT = 0.20; // 20% of image width
    private static final double DOWNSCALE_IMAGE_SIZE = 200.0;
    private static final double EXPANSION_PERCENT = 0.25; // Expand by 25% for a tighter fit

    private Point[] stableCorners = null;
    private int framesSinceLastDetection = 0;

    public DocumentDetection(Context context) {}

    public List<Point[]> detectDocumentCornersPoints(Bitmap bitmap) {
        if (bitmap == null) { return Collections.emptyList(); }

        Mat originalMat = new Mat();
        Utils.bitmapToMat(bitmap, originalMat);

        // This is the correct call to our single, robust pipeline
        Point[] newCorners = findDocumentWithKMeans(originalMat);

        // Expand the corners slightly for a better fit
        if (newCorners != null) {
            newCorners = expandCorners(newCorners, EXPANSION_PERCENT);
        }

        this.stableCorners = filterAndVerifyCorners(newCorners, originalMat.width());

        originalMat.release();

        if (this.stableCorners == null) {
            return Collections.emptyList();
        } else {
            return Collections.singletonList(this.stableCorners);
        }
    }

    private Point[] findDocumentWithKMeans(Mat image) {
        double ratio = DOWNSCALE_IMAGE_SIZE / Math.max(image.width(), image.height());
        Size downscaledSize = new Size(image.width() * ratio, image.height() * ratio);
        Mat downscaledMat = new Mat();
        Imgproc.resize(image, downscaledMat, downscaledSize, 0, 0, Imgproc.INTER_AREA);

        Mat blurredMat = new Mat();
        Imgproc.medianBlur(downscaledMat, blurredMat, 3);
        downscaledMat.release();

        Mat labImage = new Mat();
        Imgproc.cvtColor(blurredMat, labImage, Imgproc.COLOR_RGB2Lab);
        blurredMat.release();

        List<Mat> labChannels = new ArrayList<>(3);
        Core.split(labImage, labChannels);
        Mat abChannels = new Mat();
        Core.merge(Arrays.asList(labChannels.get(1), labChannels.get(2)), abChannels);
        labImage.release();
        labChannels.clear();

        Mat samples = abChannels.reshape(1, abChannels.cols() * abChannels.rows());
        Mat samples32f = new Mat();
        samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
        samples.release();
        abChannels.release();

        Mat labels = new Mat();
        Core.kmeans(samples32f, 4, labels, new org.opencv.core.TermCriteria(org.opencv.core.TermCriteria.EPS + org.opencv.core.TermCriteria.MAX_ITER, 10, 1.0), 3, Core.KMEANS_PP_CENTERS, new Mat());
        samples32f.release();

        double bestScore = -1;
        Point[] bestCorners = null;
        double minArea = downscaledSize.width * downscaledSize.height * 0.05;
        double maxArea = downscaledSize.width * downscaledSize.height * 0.95; // Critical sanity check

        for (int i = 0; i < 4; i++) {
            Mat mask = createMaskFromLabels(labels, i, downscaledSize);

            // --- CRITICAL FIX FOR "LOCK-ON-TO-FRAME" BUG ---
            // Erode the mask slightly to detach it from the image borders.
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Imgproc.erode(mask, mask, kernel);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
            kernel.release();
            // --- END OF FIX ---

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > minArea && area < maxArea) { // Check against maxArea
                    Point[] corners = getCornersFromContour(contour);
                    if (corners != null) {
                        double score = area * calculateRectangularity(contour);
                        if (score > bestScore) {
                            bestScore = score;
                            bestCorners = corners;
                        }
                    }
                }
            }
            mask.release();
        }
        labels.release();
        if (bestCorners == null) return null;

        Point[] upscaledCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            upscaledCorners[i] = new Point(bestCorners[i].x / ratio, bestCorners[i].y / ratio);
        }
        return upscaledCorners;
    }

    // --- ADVANCED JITTER FILTER ---
    private Point[] filterAndVerifyCorners(Point[] newCorners, int imageWidth) {
        if (newCorners == null) {
            framesSinceLastDetection++;
            if (framesSinceLastDetection > 5) { return null; }
            else { return stableCorners; }
        }
        framesSinceLastDetection = 0;
        if (stableCorners == null) { return newCorners; }

        double jitterThreshold = imageWidth * JITTER_REJECTION_THRESHOLD_PERCENT;
        double resetThreshold = imageWidth * MOVEMENT_RESET_THRESHOLD_PERCENT;
        double avgDistance = calculateAverageDistance(newCorners, stableCorners);

        if (avgDistance > resetThreshold) { return newCorners; }
        if (avgDistance < jitterThreshold) { return stableCorners; }

        return applyExponentialSmoothing(newCorners, stableCorners);
    }

    private Mat createMaskFromLabels(Mat labels, int clusterIndex, Size outputSize) {
        Mat mask = new Mat(outputSize, CvType.CV_8UC1);
        byte[] maskData = new byte[(int) mask.total()];
        int[] labelsData = new int[(int) labels.total()];
        labels.get(0, 0, labelsData);
        for (int j = 0; j < labelsData.length; j++) {
            if (labelsData[j] == clusterIndex) { maskData[j] = (byte) 255; }
        }
        mask.put(0, 0, maskData);
        return mask;
    }
    private MatOfPoint findLargestContour(Mat processedMat) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(processedMat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours.stream().max((c1, c2) -> Double.compare(Imgproc.contourArea(c1), Imgproc.contourArea(c2))).orElse(null);
    }
    private Point[] getCornersFromContour(MatOfPoint contour) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);
        MatOfPoint2f approx = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approx, 0.02 * perimeter, true);
        Point[] points = approx.toArray();
        contour2f.release(); approx.release();
        return (points.length == 4 && Imgproc.isContourConvex(new MatOfPoint(points))) ? sortCorners(points) : null;
    }
    private double calculateRectangularity(MatOfPoint contour) {
        if (contour == null || contour.empty()) return 0;
        RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
        double rectArea = rect.size.width * rect.size.height;
        return (rectArea == 0) ? 0 : Imgproc.contourArea(contour) / rectArea;
    }
    private Point[] applyExponentialSmoothing(Point[] current, Point[] previous) {
        Point[] smoothedCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            double x = SMOOTHING_ALPHA * current[i].x + (1 - SMOOTHING_ALPHA) * previous[i].x;
            double y = SMOOTHING_ALPHA * current[i].y + (1 - SMOOTHING_ALPHA) * previous[i].y;
            smoothedCorners[i] = new Point(x, y);
        }
        return smoothedCorners;
    }
    private double calculateAverageDistance(Point[] corners1, Point[] corners2) {
        double totalDistance = 0;
        for (int i = 0; i < 4; i++) {
            totalDistance += Math.hypot(corners1[i].x - corners2[i].x, corners1[i].y - corners2[i].y);
        }
        return totalDistance / 4.0;
    }
    private Point[] expandCorners(Point[] corners, double percentage) {
        double centerX = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
        double centerY = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;
        Point[] expandedCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            double vecX = corners[i].x - centerX;
            double vecY = corners[i].y - centerY;
            expandedCorners[i] = new Point(corners[i].x + vecX * percentage, corners[i].y + vecY * percentage);
        }
        return expandedCorners;
    }
    public Bitmap warpToDocumentFromPoints(Bitmap bitmap, Point[] points) {
        Mat src = new Mat();
        Utils.bitmapToMat(bitmap, src);
        Point[] sortedPoints = sortCorners(points);
        Point tl = sortedPoints[0], tr = sortedPoints[1], br = sortedPoints[2], bl = sortedPoints[3];
        double widthTop = Math.hypot(tr.x - tl.x, tr.y - tl.y);
        double widthBottom = Math.hypot(br.x - bl.x, br.y - bl.y);
        double maxWidth = Math.max(widthTop, widthBottom);
        double heightLeft = Math.hypot(bl.x - tl.x, bl.y - tl.y);
        double heightRight = Math.hypot(br.x - tr.x, br.y - tr.y);
        double maxHeight = Math.max(heightLeft, heightRight);
        if (maxWidth < 1 || maxHeight < 1) { src.release(); return null; }
        Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
        srcPoints.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);
        Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);
        dstPoints.put(0, 0, 0.0, 0.0, maxWidth - 1, 0.0, maxWidth - 1, maxHeight - 1, 0.0, maxHeight - 1);
        Mat warpMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Mat dst = new Mat((int) maxHeight, (int) maxWidth, src.type());
        Imgproc.warpPerspective(src, dst, warpMat, dst.size());
        Bitmap output = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dst, output);
        src.release(); dst.release(); srcPoints.release(); dstPoints.release(); warpMat.release();
        return output;
    }
    private static Point[] sortCorners(Point[] points) {
        Arrays.sort(points, (p1, p2) -> Double.compare(p1.y, p2.y));
        Point[] topPoints = {points[0], points[1]};
        Point[] bottomPoints = {points[2], points[3]};
        Arrays.sort(topPoints, (p1, p2) -> Double.compare(p1.x, p2.x));
        Arrays.sort(bottomPoints, (p1, p2) -> Double.compare(p1.x, p2.x));
        return new Point[]{topPoints[0], topPoints[1], bottomPoints[1], bottomPoints[0]};
    }
}