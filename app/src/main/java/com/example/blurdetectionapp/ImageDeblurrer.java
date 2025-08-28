package com.example.blurdetectionapp;

import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

/**
 * Advanced image deblurring class using multiple non-ML techniques
 * Optimized for small objects like rice, beans, etc.
 */
public class ImageDeblurrer {
    private static final String TAG = "ImageDeblurrer";

    // Deblurring parameters
    private static final int RICHARDSON_LUCY_ITERATIONS = 15;
    private static final int WIENER_KERNEL_SIZE = 5;
    private static final double UNSHARP_AMOUNT = 1.5;
    private static final double UNSHARP_THRESHOLD = 0.0;
    private static final int BLIND_DECONV_ITERATIONS = 10;

    public DeblurringResult deblurImage(Bitmap inputBitmap, List<BlurredObject> blurredObjects) {
        Log.d(TAG, "Starting image deblurring for " + blurredObjects.size() + " objects");

        try {
            // Convert bitmap to OpenCV Mat
            Mat inputMat = new Mat();
            Utils.bitmapToMat(inputBitmap, inputMat);

            Mat outputMat = inputMat.clone();
            List<DeblurredObject> deblurredObjects = new ArrayList<>();

            for (BlurredObject blurredObj : blurredObjects) {
                Log.d(TAG, "Deblurring object: " + blurredObj.toString());

                // Extract ROI
                Mat objectRoi = new Mat(inputMat, blurredObj.getBoundingRect());

                // Determine best deblurring method based on blur characteristics
                Mat deblurredRoi = selectAndApplyDeblurringMethod(objectRoi, blurredObj);

                // Replace the ROI in output image
                deblurredRoi.copyTo(new Mat(outputMat, blurredObj.getBoundingRect()));

                // Create deblurred object result
                DeblurredObject deblurredObj = new DeblurredObject(
                        blurredObj,
                        deblurredRoi.clone(),
                        calculateQualityImprovement(objectRoi, deblurredRoi)
                );

                deblurredObjects.add(deblurredObj);

                Log.d(TAG, "Deblurring complete for object. Quality improvement: " +
                        String.format("%.2f", deblurredObj.getQualityImprovement()));
            }

            // Convert result back to bitmap
            Bitmap outputBitmap = Bitmap.createBitmap(
                    outputMat.cols(), outputMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(outputMat, outputBitmap);

            Log.d(TAG, "Image deblurring completed successfully");
            return new DeblurringResult(outputBitmap, deblurredObjects, true);

        } catch (Exception e) {
            Log.e(TAG, "Error in image deblurring", e);
            return new DeblurringResult(inputBitmap, new ArrayList<>(), false);
        }
    }

    private Mat selectAndApplyDeblurringMethod(Mat objectRoi, BlurredObject blurredObj) {
        try {
            // Convert to grayscale for processing
            Mat grayRoi = new Mat();
            if (objectRoi.channels() > 1) {
                Imgproc.cvtColor(objectRoi, grayRoi, Imgproc.COLOR_BGR2GRAY);
            } else {
                grayRoi = objectRoi.clone();
            }

            // Select deblurring method based on blur severity and object characteristics
            Mat deblurredGray;

            if (blurredObj.getLaplacianVariance() < 50) {
                // Severe blur - use Richardson-Lucy deconvolution
                Log.d(TAG, "Applying Richardson-Lucy deconvolution for severe blur");
                deblurredGray = applyRichardsonLucy(grayRoi, blurredObj);
            } else if (blurredObj.getSobelMagnitude() < 30) {
                // Moderate blur - use Wiener filtering
                Log.d(TAG, "Applying Wiener filtering for moderate blur");
                deblurredGray = applyWienerFilter(grayRoi, blurredObj);
            } else {
                // Mild blur - use unsharp masking
                Log.d(TAG, "Applying unsharp masking for mild blur");
                deblurredGray = applyUnsharpMasking(grayRoi);
            }

            // Apply additional enhancement
            Mat enhanced = applyAdditionalEnhancement(deblurredGray, blurredObj);

            // Convert back to original color format if needed
            Mat result = new Mat();
            if (objectRoi.channels() > 1) {
                // For color images, apply enhancement to each channel
                List<Mat> channels = new ArrayList<>();
                Core.split(objectRoi, channels);

                for (int i = 0; i < channels.size(); i++) {
                    Mat channelEnhanced = applyChannelEnhancement(channels.get(i), enhanced);
                    channels.set(i, channelEnhanced);
                }

                Core.merge(channels, result);
            } else {
                result = enhanced;
            }

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error in deblurring method selection", e);
            return objectRoi.clone();
        }
    }

    private Mat applyRichardsonLucy(Mat grayRoi, BlurredObject blurredObj) {
        try {
            // Estimate PSF (Point Spread Function) based on object characteristics
            Mat psf = estimatePSF(blurredObj);

            // Convert to floating point
            Mat image = new Mat();
            grayRoi.convertTo(image, CvType.CV_64F, 1.0/255.0);

            Mat result = image.clone();
            Mat psfFlipped = new Mat();
            Core.flip(psf, psfFlipped, -1);

            // Richardson-Lucy iterations
            for (int i = 0; i < RICHARDSON_LUCY_ITERATIONS; i++) {
                // Convolve result with PSF
                Mat convolved = new Mat();
                Imgproc.filter2D(result, convolved, -1, psf);

                // Avoid division by zero
                Mat safeDivisor = new Mat();
                Core.max(convolved, new Scalar(1e-10), safeDivisor);

                // Calculate ratio
                Mat ratio = new Mat();
                Core.divide(image, safeDivisor, ratio);

                // Convolve ratio with flipped PSF
                Mat correction = new Mat();
                Imgproc.filter2D(ratio, correction, -1, psfFlipped);

                // Update result
                Core.multiply(result, correction, result);
            }

            // Convert back to 8-bit
            Mat output = new Mat();
            Core.multiply(result, new Scalar(255.0), result);
            result.convertTo(output, CvType.CV_8U);

            return output;

        } catch (Exception e) {
            Log.e(TAG, "Error in Richardson-Lucy deconvolution", e);
            return grayRoi.clone();
        }
    }

    private Mat applyWienerFilter(Mat grayRoi, BlurredObject blurredObj) {
        try {
            // Estimate noise-to-signal ratio based on blur metrics
            double nsr = estimateNoiseToSignalRatio(blurredObj);

            // Apply DFT
            Mat padded = new Mat();
            int m = Core.getOptimalDFTSize(grayRoi.rows());
            int n = Core.getOptimalDFTSize(grayRoi.cols());
            Core.copyMakeBorder(grayRoi, padded, 0, m - grayRoi.rows(), 0, n - grayRoi.cols(),
                    Core.BORDER_CONSTANT, Scalar.all(0));

            Mat[] planes = {new Mat(), new Mat()};
            padded.convertTo(planes[0], CvType.CV_64F);
            planes[1] = Mat.zeros(padded.size(), CvType.CV_64F);

            Mat complexImage = new Mat();
            Core.merge(new ArrayList<>(java.util.Arrays.asList(planes)), complexImage);

            Core.dft(complexImage, complexImage);

            // Create Wiener filter
            Mat wienerFilter = createWienerFilter(padded.size(), blurredObj, nsr);

            // Apply filter in frequency domain
            Mat filtered = new Mat();
            Core.mulSpectrums(complexImage, wienerFilter, filtered, 0);

            // Inverse DFT
            Core.idft(filtered, filtered);

            // Extract real part
            List<Mat> resultPlanes = new ArrayList<>();
            Core.split(filtered, resultPlanes);

            Mat result = new Mat();
            resultPlanes.get(0).convertTo(result, CvType.CV_8U);

            // Crop to original size
            return new Mat(result, new Rect(0, 0, grayRoi.cols(), grayRoi.rows()));

        } catch (Exception e) {
            Log.e(TAG, "Error in Wiener filtering", e);
            return applyUnsharpMasking(grayRoi); // Fallback to unsharp masking
        }
    }

    private Mat applyUnsharpMasking(Mat grayRoi) {
        try {
            // Create Gaussian blur
            Mat blurred = new Mat();
            Size ksize = new Size(WIENER_KERNEL_SIZE, WIENER_KERNEL_SIZE);
            Imgproc.GaussianBlur(grayRoi, blurred, ksize, 0);

            // Create mask (original - blurred)
            Mat mask = new Mat();
            Core.subtract(grayRoi, blurred, mask);

            // Apply unsharp masking: original + amount * mask
            Mat result = new Mat();
            Core.addWeighted(grayRoi, 1.0, mask, UNSHARP_AMOUNT, 0, result);

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error in unsharp masking", e);
            return grayRoi.clone();
        }
    }

    private Mat applyAdditionalEnhancement(Mat deblurred, BlurredObject blurredObj) {
        try {
            Mat enhanced = deblurred.clone();

            // Apply bilateral filter to reduce noise while preserving edges
            Mat bilateralFiltered = new Mat();
            Imgproc.bilateralFilter(enhanced, bilateralFiltered, 5, 50, 50);

            // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);
            clahe.setTilesGridSize(new Size(4, 4));

            Mat claheEnhanced = new Mat();
            clahe.apply(bilateralFiltered, claheEnhanced);

            // Apply edge enhancement if object has low edge density
            if (blurredObj.getTenengrad() < 1000) {
                Mat edgeEnhanced = enhanceEdges(claheEnhanced);
                return edgeEnhanced;
            }

            return claheEnhanced;

        } catch (Exception e) {
            Log.e(TAG, "Error in additional enhancement", e);
            return deblurred;
        }
    }

    private Mat enhanceEdges(Mat input) {
        try {
            // Calculate edges using Sobel operator
            Mat sobelX = new Mat();
            Mat sobelY = new Mat();
            Imgproc.Sobel(input, sobelX, CvType.CV_64F, 1, 0, 3);
            Imgproc.Sobel(input, sobelY, CvType.CV_64F, 0, 1, 3);

            // Calculate magnitude
            Mat magnitude = new Mat();
            Core.magnitude(sobelX, sobelY, magnitude);

            // Normalize and convert back to 8-bit
            Core.normalize(magnitude, magnitude, 0, 255, Core.NORM_MINMAX);
            magnitude.convertTo(magnitude, CvType.CV_8U);

            // Add edge information back to original
            Mat result = new Mat();
            Core.addWeighted(input, 0.8, magnitude, 0.2, 0, result);

            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error in edge enhancement", e);
            return input;
        }
    }

    private Mat applyChannelEnhancement(Mat channel, Mat reference) {
        try {
            // Apply histogram matching to align channel with reference
            Mat enhanced = new Mat();

            // Calculate histograms
            Mat histChannel = new Mat();
            Mat histReference = new Mat();

            Imgproc.calcHist(java.util.Arrays.asList(channel), new MatOfInt(0),
                    new Mat(), histChannel, new MatOfInt(256), new MatOfFloat(0, 256));
            Imgproc.calcHist(java.util.Arrays.asList(reference), new MatOfInt(0),
                    new Mat(), histReference, new MatOfInt(256), new MatOfFloat(0, 256));

            // Create lookup table for histogram matching
            Mat lut = createHistogramMatchingLUT(histChannel, histReference);

            // Apply LUT
            Core.LUT(channel, lut, enhanced);

            return enhanced;

        } catch (Exception e) {
            Log.e(TAG, "Error in channel enhancement", e);
            return channel.clone();
        }
    }

    private Mat createHistogramMatchingLUT(Mat srcHist, Mat refHist) {
        try {
            // Calculate cumulative distribution functions
            Mat srcCDF = calculateCDF(srcHist);
            Mat refCDF = calculateCDF(refHist);

            // Create lookup table
            Mat lut = new Mat(256, 1, CvType.CV_8U);

            for (int i = 0; i < 256; i++) {
                double srcValue = srcCDF.get(i, 0)[0];

                // Find closest value in reference CDF
                int matchIndex = 0;
                double minDiff = Double.MAX_VALUE;

                for (int j = 0; j < 256; j++) {
                    double refValue = refCDF.get(j, 0)[0];
                    double diff = Math.abs(srcValue - refValue);

                    if (diff < minDiff) {
                        minDiff = diff;
                        matchIndex = j;
                    }
                }

                lut.put(i, 0, matchIndex);
            }

            return lut;

        } catch (Exception e) {
            Log.e(TAG, "Error creating histogram matching LUT", e);
            return Mat.eye(256, 1, CvType.CV_8U);
        }
    }

    private Mat calculateCDF(Mat histogram) {
        Mat cdf = new Mat(histogram.size(), histogram.type());
        double sum = 0;
        double total = Core.sumElems(histogram).val[0];

        for (int i = 0; i < histogram.rows(); i++) {
            sum += histogram.get(i, 0)[0];
            cdf.put(i, 0, sum / total);
        }

        return cdf;
    }

    private Mat estimatePSF(BlurredObject blurredObj) {
        try {
            // Estimate PSF size based on object size and blur characteristics
            int psfSize = Math.max(3, Math.min(15, (int) Math.sqrt(blurredObj.getArea()) / 10));
            if (psfSize % 2 == 0) psfSize++; // Ensure odd size

            // Create Gaussian PSF based on estimated blur parameters
            double sigma = estimateBlurSigma(blurredObj);

            Mat psf = new Mat(psfSize, psfSize, CvType.CV_64F);

            int center = psfSize / 2;
            double sum = 0;

            for (int y = 0; y < psfSize; y++) {
                for (int x = 0; x < psfSize; x++) {
                    double dx = x - center;
                    double dy = y - center;
                    double value = Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                    psf.put(y, x, value);
                    sum += value;
                }
            }

            // Normalize PSF
            Core.multiply(psf, new Scalar(1.0 / sum), psf);

            return psf;

        } catch (Exception e) {
            Log.e(TAG, "Error estimating PSF", e);
            // Return default 5x5 Gaussian kernel
            return Imgproc.getGaussianKernel(5, 1.0);
        }
    }

    private double estimateBlurSigma(BlurredObject blurredObj) {
        // Estimate blur sigma based on Laplacian variance
        // Lower variance indicates more blur, higher sigma needed
        double normalizedVariance = Math.max(1, blurredObj.getLaplacianVariance()) / 100.0;
        return Math.max(0.5, Math.min(3.0, 2.0 / normalizedVariance));
    }

    private double estimateNoiseToSignalRatio(BlurredObject blurredObj) {
        // Estimate NSR based on pixel variance and edge density
        double pixelVar = blurredObj.getPixelVariance();
        double edgeDensity = blurredObj.getTenengrad();

        // Higher pixel variance and lower edge density indicate more noise
        double nsr = pixelVar / Math.max(1, edgeDensity / 1000.0);
        return Math.max(0.001, Math.min(0.1, nsr / 10000.0));
    }

    private Mat createWienerFilter(Size size, BlurredObject blurredObj, double nsr) {
        try {
            // Create PSF in frequency domain
            Mat psf = estimatePSF(blurredObj);

            // Pad PSF to match image size
            Mat paddedPSF = new Mat();
            int padX = (int) (size.width - psf.cols());
            int padY = (int) (size.height - psf.rows());
            Core.copyMakeBorder(psf, paddedPSF, 0, padY, 0, padX, Core.BORDER_CONSTANT, Scalar.all(0));

            // Convert to complex and apply DFT
            Mat[] psfPlanes = {new Mat(), new Mat()};
            paddedPSF.convertTo(psfPlanes[0], CvType.CV_64F);
            psfPlanes[1] = Mat.zeros(paddedPSF.size(), CvType.CV_64F);

            Mat complexPSF = new Mat();
            Core.merge(java.util.Arrays.asList(psfPlanes), complexPSF);
            Core.dft(complexPSF, complexPSF);

            // Create Wiener filter: conj(H) / (|H|^2 + NSR)
            Mat psfConj = new Mat();
            Mat psfMagnitudeSquared = new Mat();

            // Calculate conjugate
            List<Mat> psfChannels = new ArrayList<>();
            Core.split(complexPSF, psfChannels);
            Core.multiply(psfChannels.get(1), new Scalar(-1), psfChannels.get(1)); // Negate imaginary part
            Core.merge(psfChannels, psfConj);

            // Calculate |H|^2
            List<Mat> magChannels = new ArrayList<>();
            Core.split(complexPSF, magChannels);
            Mat realSquared = new Mat();
            Mat imagSquared = new Mat();
            Core.multiply(magChannels.get(0), magChannels.get(0), realSquared);
            Core.multiply(magChannels.get(1), magChannels.get(1), imagSquared);
            Core.add(realSquared, imagSquared, psfMagnitudeSquared);

            // Add NSR
            Mat denominator = new Mat();
            Core.add(psfMagnitudeSquared, new Scalar(nsr), denominator);

            // Create Wiener filter
            Mat wienerFilter = new Mat();
            List<Mat> wienerChannels = new ArrayList<>();
            Core.split(psfConj, wienerChannels);

            Mat realFiltered = new Mat();
            Mat imagFiltered = new Mat();
            Core.divide(wienerChannels.get(0), denominator, realFiltered);
            Core.divide(wienerChannels.get(1), denominator, imagFiltered);

            wienerChannels.set(0, realFiltered);
            wienerChannels.set(1, imagFiltered);
            Core.merge(wienerChannels, wienerFilter);

            return wienerFilter;

        } catch (Exception e) {
            Log.e(TAG, "Error creating Wiener filter", e);
            // Return identity filter as fallback
            Mat identity = Mat.zeros(size, CvType.CV_64FC2);
            identity.put(0, 0, 1.0, 0.0);
            return identity;
        }
    }

    private double calculateQualityImprovement(Mat original, Mat deblurred) {
        try {
            // Calculate improvement based on edge enhancement
            double originalEdges = calculateEdgeStrength(original);
            double deblurredEdges = calculateEdgeStrength(deblurred);

            double improvement = (deblurredEdges - originalEdges) / Math.max(1, originalEdges);
            return Math.max(0, improvement * 100); // Return as percentage

        } catch (Exception e) {
            Log.e(TAG, "Error calculating quality improvement", e);
            return 0.0;
        }
    }

    private double calculateEdgeStrength(Mat image) {
        try {
            Mat sobelX = new Mat();
            Mat sobelY = new Mat();
            Mat magnitude = new Mat();

            Imgproc.Sobel(image, sobelX, CvType.CV_64F, 1, 0, 3);
            Imgproc.Sobel(image, sobelY, CvType.CV_64F, 0, 1, 3);
            Core.magnitude(sobelX, sobelY, magnitude);

            Scalar meanMagnitude = Core.mean(magnitude);
            return meanMagnitude.val[0];

        } catch (Exception e) {
            Log.e(TAG, "Error calculating edge strength", e);
            return 0.0;
        }
    }
}