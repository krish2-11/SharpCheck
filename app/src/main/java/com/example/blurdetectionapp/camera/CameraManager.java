package com.example.blurdetectionapp.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.example.blurdetectionapp.utils.BlurDetector;
import com.example.blurdetectionapp.utils.LightingAnalyzer;
import com.example.blurdetectionapp.utils.OverlayView;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Unified Camera Manager using CameraX with OpenCV support
 * Combines preview, capture, analysis, and ROI extraction capabilities
 */
@ExperimentalGetImage
public class CameraManager {

    private static final String TAG = "CameraManager";

    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final ExecutorService cameraExecutor;

    private ProcessCameraProvider cameraProvider;
    private Camera camera;
    private Preview preview;
    private ImageCapture imageCapture;
    private ImageAnalysis imageAnalysis;

    // Reference to overlay view for ROI extraction
    private OverlayView overlayView;

    // Callbacks
    private LightingAnalysisCallback lightingCallback;
    private BlurAnalysisCallback blurCallback;
    private ImageCaptureCallback captureCallback;
    private FrameCallback frameCallback;

    // Callback Interfaces
    public interface LightingAnalysisCallback {
        void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result);
    }

    public interface BlurAnalysisCallback {
        void onBlurAnalyzed(BlurDetector.BlurDetectionResult result);
    }

    public interface ImageCaptureCallback {
        void onImageCaptured(Bitmap bitmap);
        void onCaptureError(String error);
    }

    public interface FrameCallback {
        void onFrameAvailable(Bitmap bitmap);
    }

    public CameraManager(Context context, LifecycleOwner lifecycleOwner) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.cameraExecutor = Executors.newSingleThreadExecutor(); // Allow parallel analysis
    }

    /**
     * Set overlay view reference for ROI extraction
     */
    public void setOverlayView(OverlayView overlayView) {
        this.overlayView = overlayView;
    }

    /**
     * Set frame callback for live frame processing
     */
    public void setFrameCallback(FrameCallback callback) {
        this.frameCallback = callback;
    }

    /**
     * Initialize camera with all use cases
     */
    public void initializeCamera(PreviewView previewView,
                                 LightingAnalysisCallback lightingCallback,
                                 BlurAnalysisCallback blurCallback,
                                 ImageCaptureCallback captureCallback) {
        this.lightingCallback = lightingCallback;
        this.blurCallback = blurCallback;
        this.captureCallback = captureCallback;

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(context);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                startCamera(previewView);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Failed to get camera provider", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    /**
     * Start camera with all use cases
     */
    private void startCamera(PreviewView previewView) {
        // Preview use case
        preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Image capture use case
        imageCapture = new ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build();

        // Image analysis use case
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480)) // Lower resolution for faster analysis
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, new CombinedImageAnalyzer());

        // Camera selector
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        try {
            // Unbind any existing use cases
            cameraProvider.unbindAll();

            // Bind use cases to camera
            camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageCapture,
                    imageAnalysis
            );

            Log.d(TAG, "Camera started successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to start camera", e);
        }
    }

    /**
     * Extract ROI from Mat based on overlay view coordinates
     */
    private Mat extractROIFromMat(Mat mat) {
        if (overlayView == null) {
            return mat; // Fallback to full mat
        }

        RectF overlayRect = overlayView.getOverlayRect();
        if (overlayRect == null) {
            return mat; // Fallback to full mat
        }

        // Convert overlay rect from view coordinates to mat coordinates
        int viewWidth = overlayView.getWidth();
        int viewHeight = overlayView.getHeight();

        if (viewWidth == 0 || viewHeight == 0) {
            return mat; // Fallback if view not measured
        }

        // Scale overlay rect to mat dimensions
        float scaleX = (float) mat.cols() / viewWidth;
        float scaleY = (float) mat.rows() / viewHeight;

        int left = Math.max(0, Math.round(overlayRect.left * scaleX));
        int top = Math.max(0, Math.round(overlayRect.top * scaleY));
        int right = Math.min(mat.cols(), Math.round(overlayRect.right * scaleX));
        int bottom = Math.min(mat.rows(), Math.round(overlayRect.bottom * scaleY));

        // Ensure valid rect
        if (right <= left || bottom <= top) {
            return mat; // Fallback to full mat
        }

        try {
            // Extract ROI from mat
            org.opencv.core.Rect roiRect = new org.opencv.core.Rect(left, top, right - left, bottom - top);
            return new Mat(mat, roiRect);
        } catch (Exception e) {
            Log.e(TAG, "Error extracting ROI from Mat: " + e.getMessage());
            return mat; // Fallback to full mat
        }
    }

    /**
     * Convert Mat to Bitmap for ROI extraction in MainActivity
     */
    private Bitmap matToROIBitmap(Mat roiMat) {
        if (roiMat == null) return null;
        try {
            Bitmap bitmap = Bitmap.createBitmap(roiMat.cols(), roiMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(roiMat, bitmap);
            return bitmap;
        } catch (Exception e) {
            Log.e(TAG, "Error converting ROI Mat to Bitmap", e);
            return null;
        }
    }

    /**
     * Combined Image Analyzer for all analysis tasks
     */
    private class CombinedImageAnalyzer implements ImageAnalysis.Analyzer {

        @Override
        public void analyze(@NonNull ImageProxy image) {
            long currentTime = System.currentTimeMillis();

            // Convert ImageProxy to OpenCV Mat
            Mat mat = imageProxyToMat(image);
            if (mat == null) {
                image.close();
                return;
            }

            // Frame callback (use full frame for overlay mapping)
            if (frameCallback != null) {
                Bitmap bitmap = matToBitmap(mat);
                if (bitmap != null) {
                    ContextCompat.getMainExecutor(context).execute(() ->
                            frameCallback.onFrameAvailable(bitmap)
                    );
                }
            }

            // Extract ROI for analysis
            Mat roiMat = extractROIFromMat(mat);

            // Lighting analysis on the FULL frame
            if (lightingCallback != null) {
                Mat lightingMat = roiMat.clone(); // Use mat
                cameraExecutor.execute(() -> {
                    LightingAnalyzer.LightingAnalysisResult result =
                            LightingAnalyzer.analyzeLighting(lightingMat);
                    lightingMat.release();

                    ContextCompat.getMainExecutor(context).execute(() ->

                            lightingCallback.onLightingAnalyzed(result)

                    );
                });
            }

            // Blur analysis on the FULL frame
            if (blurCallback != null) {
                Mat blurMat = roiMat.clone(); // Use mat
                cameraExecutor.execute(() -> {
                    BlurDetector.BlurDetectionResult result =
                            BlurDetector.detectBlurAndOcclusion(blurMat);
                    blurMat.release();

                    ContextCompat.getMainExecutor(context).execute(() ->

                            blurCallback.onBlurAnalyzed(result)

                    );
                });
            }

            // Clean up
            if (roiMat != mat) {
                roiMat.release();
            }
            mat.release();
            image.close();
        }
    }

    /**
     * Convert OpenCV Mat to Bitmap
     */
    private Bitmap matToBitmap(Mat mat) {
        try {
            Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap);
            return bitmap;
        } catch (Exception e) {
            Log.e(TAG, "Error converting Mat to Bitmap", e);
            return null;
        }
    }

    /**
     * Convert ImageProxy to OpenCV Mat
     */
    private Mat imageProxyToMat(ImageProxy image) {
        Image mediaImage = image.getImage();
        if (mediaImage == null) return null;

        try {
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();

            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuv.put(0, 0, nv21);

            Mat rgb = new Mat();
            Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21, 3);
            yuv.release();
            return rgb;
        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Mat", e);
            return null;
        }
    }

    /**
     * Convert ImageProxy to Bitmap (for capture)
     */
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        Image mediaImage = image.getImage();
        if (mediaImage == null) {
            Log.e(TAG, "MediaImage is null, cannot convert to Bitmap");
            return null;
        }

        int format = mediaImage.getFormat();

        try {
            if (format == ImageFormat.YUV_420_888) {
                Image.Plane[] planes = mediaImage.getPlanes();
                if (planes.length < 3) {
                    Log.e(TAG, "YUV_420_888 image does not have 3 planes");
                    return null;
                }

                ByteBuffer yBuffer = planes[0].getBuffer();
                ByteBuffer uBuffer = planes[1].getBuffer();
                ByteBuffer vBuffer = planes[2].getBuffer();

                int ySize = yBuffer.remaining();
                int uSize = uBuffer.remaining();
                int vSize = vBuffer.remaining();

                byte[] nv21 = new byte[ySize + uSize + vSize];

                yBuffer.get(nv21, 0, ySize);
                vBuffer.get(nv21, ySize, vSize);
                uBuffer.get(nv21, ySize + vSize, uSize);

                YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21,
                        image.getWidth(), image.getHeight(), null);
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
                byte[] imageBytes = out.toByteArray();

                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

            } else if (format == ImageFormat.JPEG) {
                ByteBuffer buffer = mediaImage.getPlanes()[0].getBuffer();
                byte[] jpegBytes = new byte[buffer.remaining()];
                buffer.get(jpegBytes);
                return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
            } else {
                Log.e(TAG, "Unsupported image format: " + format);
                return null;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e);
            return null;
        }
    }

    /**
     * Rotate bitmap based on rotation degrees
     */
    private Bitmap rotateBitmap(Bitmap source, float angle) {
        if (angle == 0) return source;

        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private Mat rotateMat(Mat source, double angle) {
        if (angle == 0) return source;

        // Get image center
        Point center = new Point(source.cols() / 2.0, source.rows() / 2.0);

        // Calculate rotation matrix
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        // Destination mat
        Mat rotated = new Mat();

        // Perform the affine warp (rotation)
        Imgproc.warpAffine(source, rotated, rotationMatrix, source.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT);

        return rotated;
    }


    /**
     * Capture image to bitmap
     */
    public void captureImage() {
        if (imageCapture == null) {
            if (captureCallback != null) {
                captureCallback.onCaptureError("Camera not initialized");
            }
            return;
        }

        imageCapture.takePicture(
                ContextCompat.getMainExecutor(context),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        Bitmap bitmap = imageProxyToBitmap(image);

                        // Fix orientation
                        int rotation = image.getImageInfo().getRotationDegrees();
                        if (bitmap != null && rotation != 0) {
                            bitmap = rotateBitmap(bitmap, rotation);
                        }

                        image.close();

                        if (captureCallback != null && bitmap != null) {
                            captureCallback.onImageCaptured(bitmap);
                        }
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Capture failed", exception);
                        if (captureCallback != null) {
                            captureCallback.onCaptureError("Capture failed: " + exception.getMessage());
                        }
                    }
                }
        );
    }

    /**
     * Check if camera is available
     */
    public boolean isCameraAvailable() {
        return camera != null;
    }

    /**
     * Enable/disable torch
     */
    public void setTorchEnabled(boolean enabled) {
        if (camera != null && camera.getCameraInfo().hasFlashUnit()) {
            camera.getCameraControl().enableTorch(enabled);
        }
    }

    /**
     * Shutdown camera and cleanup resources
     */
    public void shutdown() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        cameraExecutor.shutdownNow();
    }
}