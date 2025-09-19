package com.example.blurdetectionapp.utils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Enhanced ShadowDetector specifically designed for document images.
 * Uses adaptive thresholding and multiple color space analysis to distinguish
 * shadows from text and other dark objects on documents.
 */
public class ShadowDetector {

    public static class ShadowDetectionResult {
        public Mat shadowMask; // Single channel mask: 255 = shadow, 0 = non-shadow
        public int shadowPixelCount;
        public int totalPixelCount;

        public ShadowDetectionResult(Mat mask) {
            this.shadowMask = mask;
            this.shadowPixelCount = Core.countNonZero(mask);
            this.totalPixelCount = mask.rows() * mask.cols();
        }

        public double getShadowRatio() {
            return totalPixelCount == 0 ? 0 : (double) shadowPixelCount / totalPixelCount;
        }
    }

    /**
     * Detect shadows in document images using adaptive multi-criteria approach.
     * @param rgbMat Input image in RGB color space.
     * @return ShadowDetectionResult containing shadow mask and stats.
     */
    public static ShadowDetectionResult detectShadows(Mat rgbMat) {
        if (rgbMat == null || rgbMat.empty()) {
            return new ShadowDetectionResult(new Mat());
        }

        // Convert to different color spaces for analysis
        Mat hsvMat = new Mat();
        Mat labMat = new Mat();
        Mat grayMat = new Mat();

        Imgproc.cvtColor(rgbMat, hsvMat, Imgproc.COLOR_RGB2HSV);
        Imgproc.cvtColor(rgbMat, labMat, Imgproc.COLOR_RGB2Lab);
        Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        // Calculate adaptive thresholds based on image statistics
        Scalar meanHSV = Core.mean(hsvMat);
        org.opencv.core.MatOfDouble meanHSVMat = new org.opencv.core.MatOfDouble();
        org.opencv.core.MatOfDouble stdDevHSVMat = new org.opencv.core.MatOfDouble();
        Core.meanStdDev(hsvMat, meanHSVMat, stdDevHSVMat);
        double[] stdDevHSVArray = stdDevHSVMat.toArray();

        Scalar meanGray = Core.mean(grayMat);

        // Split channels
        java.util.List<Mat> hsvChannels = new java.util.ArrayList<>();
        java.util.List<Mat> labChannels = new java.util.ArrayList<>();
        Core.split(hsvMat, hsvChannels);
        Core.split(labMat, labChannels);

        Mat h = hsvChannels.get(0);
        Mat s = hsvChannels.get(1);
        Mat v = hsvChannels.get(2);
        Mat l = labChannels.get(0);
        Mat a = labChannels.get(1);
        Mat b = labChannels.get(2);

        // Adaptive thresholds
        double vThreshold = Math.max(30, meanHSV.val[2] - stdDevHSVArray[2] * 0.8);
        double lThreshold = Math.max(20, meanGray.val[0] * 0.6);
        double sMinThreshold = 15; // Minimum saturation to avoid pure grays
        double sMaxThreshold = Math.min(180, meanHSV.val[1] + stdDevHSVArray[1] * 0.5);

        // 1. Value (brightness) criterion - shadows are darker
        Mat valueMask = new Mat();
        Imgproc.threshold(v, valueMask, vThreshold, 255, Imgproc.THRESH_BINARY_INV);

        // 2. Lightness criterion from LAB space
        Mat lightnessMask = new Mat();
        Imgproc.threshold(l, lightnessMask, lThreshold, 255, Imgproc.THRESH_BINARY_INV);

        // 3. Saturation criterion - shadows retain some color, pure black text doesn't
        Mat satMaskMin = new Mat();
        Mat satMaskMax = new Mat();
        Imgproc.threshold(s, satMaskMin, sMinThreshold, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(s, satMaskMax, sMaxThreshold, 255, Imgproc.THRESH_BINARY_INV);

        Mat satMask = new Mat();
        Core.bitwise_and(satMaskMin, satMaskMax, satMask);

        // 4. Color consistency check in LAB space
        // Shadows typically have neutral a and b values (close to 128)
        Mat aMask = new Mat();
        Mat bMask = new Mat();
        Core.absdiff(a, new Scalar(128), aMask);
        Core.absdiff(b, new Scalar(128), bMask);

        Mat aThresh = new Mat();
        Mat bThresh = new Mat();
        Imgproc.threshold(aMask, aThresh, 30, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.threshold(bMask, bThresh, 30, 255, Imgproc.THRESH_BINARY_INV);

        Mat colorMask = new Mat();
        Core.bitwise_and(aThresh, bThresh, colorMask);

        // 5. Edge-based exclusion to avoid text
        Mat edges = new Mat();
        Imgproc.Canny(grayMat, edges, 50, 150);

        // Dilate edges to create exclusion zone
        Mat edgeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.dilate(edges, edges, edgeKernel, new org.opencv.core.Point(-1, -1), 2);

        // Invert edges for exclusion
        Mat edgeExclusion = new Mat();
        Core.bitwise_not(edges, edgeExclusion);

        // Combine all criteria
        Mat shadowMask = new Mat();
        Core.bitwise_and(valueMask, lightnessMask, shadowMask);
        Core.bitwise_and(shadowMask, satMask, shadowMask);
        Core.bitwise_and(shadowMask, colorMask, shadowMask);
        Core.bitwise_and(shadowMask, edgeExclusion, shadowMask);

        // Morphological operations for noise reduction and shadow region completion
        Mat morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));

        // Opening to remove small noise
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_OPEN, morphKernel);

        // Closing to fill small gaps in shadow regions
        Mat closingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_CLOSE, closingKernel);

        // Final filtering based on connected components size
        Mat finalMask = filterByConnectedComponents(shadowMask, rgbMat.total() * 0.01, rgbMat.total() * 0.3);

        // Release temporary matrices
        hsvMat.release();
        labMat.release();
        grayMat.release();
        meanHSVMat.release();
        stdDevHSVMat.release();
        h.release();
        s.release();
        v.release();
        l.release();
        a.release();
        b.release();
        valueMask.release();
        lightnessMask.release();
        satMaskMin.release();
        satMaskMax.release();
        satMask.release();
        aMask.release();
        bMask.release();
        aThresh.release();
        bThresh.release();
        colorMask.release();
        edges.release();
        edgeKernel.release();
        edgeExclusion.release();
        shadowMask.release();
        morphKernel.release();
        closingKernel.release();

        return new ShadowDetectionResult(finalMask);
    }

    /**
     * Filter connected components based on size to remove noise and very large false positives.
     */
    private static Mat filterByConnectedComponents(Mat mask, double minSize, double maxSize) {
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();

        int numComponents = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S);

        Mat filteredMask = Mat.zeros(mask.size(), CvType.CV_8UC1);

        for (int i = 1; i < numComponents; i++) { // Skip background (label 0)
            double[] statsRow = stats.get(i, 0);
            double area = statsRow[4]; // CC_STAT_AREA

            if (area >= minSize && area <= maxSize) {
                // Keep this component
                Mat componentMask = new Mat();
                Core.compare(labels, new Scalar(i), componentMask, Core.CMP_EQ);
                Core.bitwise_or(filteredMask, componentMask, filteredMask);
                componentMask.release();
            }
        }

        labels.release();
        stats.release();
        centroids.release();

        return filteredMask;
    }

    /**
     * Alternative faster method with simplified criteria for real-time processing.
     */
    public static ShadowDetectionResult detectShadowsFast(Mat rgbMat) {
        if (rgbMat == null || rgbMat.empty()) {
            return new ShadowDetectionResult(new Mat());
        }

        Mat hsvMat = new Mat();
        Mat grayMat = new Mat();

        Imgproc.cvtColor(rgbMat, hsvMat, Imgproc.COLOR_RGB2HSV);
        Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        // Adaptive threshold based on image mean
        Scalar meanGray = Core.mean(grayMat);
        double threshold = meanGray.val[0] * 0.7;

        // Split HSV
        java.util.List<Mat> hsvChannels = new java.util.ArrayList<>();
        Core.split(hsvMat, hsvChannels);
        Mat s = hsvChannels.get(1);
        Mat v = hsvChannels.get(2);

        // Simple shadow detection
        Mat valueMask = new Mat();
        Mat satMask = new Mat();

        Imgproc.threshold(v, valueMask, threshold, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.threshold(s, satMask, 20, 255, Imgproc.THRESH_BINARY);

        Mat shadowMask = new Mat();
        Core.bitwise_and(valueMask, satMask, shadowMask);

        // Quick morphological cleanup
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Imgproc.morphologyEx(shadowMask, shadowMask, Imgproc.MORPH_OPEN, kernel);

        // Release resources
        hsvMat.release();
        grayMat.release();
        s.release();
        v.release();
        valueMask.release();
        satMask.release();
        kernel.release();

        return new ShadowDetectionResult(shadowMask);
    }
}