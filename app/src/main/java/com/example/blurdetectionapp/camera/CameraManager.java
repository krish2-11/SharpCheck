package com.example.blurdetectionapp.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
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

import com.example.blurdetectionapp.utils.LightingAnalyzer;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Camera manager for handling CameraX operations and live lighting analysis
 */
@androidx.camera.core.ExperimentalGetImage
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

    // Callbacks
    private LightingAnalysisCallback lightingCallback;
    private ImageCaptureCallback captureCallback;

    public interface LightingAnalysisCallback {
        void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result);
    }

    public interface ImageCaptureCallback {
        void onImageCaptured(Bitmap bitmap);
        void onCaptureError(String error);
    }

    public CameraManager(Context context, LifecycleOwner lifecycleOwner) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.cameraExecutor = Executors.newSingleThreadExecutor();
    }

    /**
     * Initialize camera with preview and analysis
     */
    public void initializeCamera(PreviewView previewView,
                                 LightingAnalysisCallback lightingCallback,
                                 ImageCaptureCallback captureCallback) {
        this.lightingCallback = lightingCallback;
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
     * Start camera with preview, capture, and analysis use cases
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

        // Image analysis use case for live lighting detection
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480)) // Lower resolution for faster analysis
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888) // ADD THIS
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, new LightingImageAnalyzer());

        // Camera selector (back camera)
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
     * Capture image
     */
    public void captureImage() {
        if (imageCapture == null) {
            if (captureCallback != null) {
                captureCallback.onCaptureError("Camera not initialized");
            }
            return;
        }

        ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(
                new java.io.File(context.getExternalFilesDir(null), "captured_image_" + System.currentTimeMillis() + ".jpg")
        ).build();

        imageCapture.takePicture(
                outputFileOptions,
                ContextCompat.getMainExecutor(context),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults output) {
                        // Also capture in memory for immediate processing
                        captureInMemory();
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Photo capture failed", exception);
                        if (captureCallback != null) {
                            captureCallback.onCaptureError("Capture failed: " + exception.getMessage());
                        }
                    }
                }
        );
    }

    /**
     * Capture image in memory for immediate processing
     */
    private void captureInMemory() {
        if (imageCapture == null) return;

        imageCapture.takePicture(
                ContextCompat.getMainExecutor(context),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        // Convert ImageProxy to Bitmap
                        Bitmap bitmap = imageProxyToBitmap(image);
                        image.close();

                        if (captureCallback != null && bitmap != null) {
                            captureCallback.onImageCaptured(bitmap);
                        }
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "In-memory capture failed", exception);
                        if (captureCallback != null) {
                            captureCallback.onCaptureError("Capture failed: " + exception.getMessage());
                        }
                    }
                }
        );
    }

    /**
     * Convert ImageProxy to Bitmap
     */
    @androidx.camera.core.ExperimentalGetImage
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
     * Image analyzer for live lighting analysis
     */
    private class LightingImageAnalyzer implements ImageAnalysis.Analyzer {
        private long lastAnalysisTime = 0;
        private static final long ANALYSIS_INTERVAL = 500; // Analyze every 500ms

        @Override
        public void analyze(@NonNull ImageProxy image) {
            long currentTime = System.currentTimeMillis();

            // Throttle analysis to avoid excessive processing
            if (currentTime - lastAnalysisTime < ANALYSIS_INTERVAL) {
                image.close();
                return;
            }

            lastAnalysisTime = currentTime;

            try {
                // Convert to bitmap for analysis
                Bitmap bitmap = imageProxyToBitmap(image);

                if (bitmap != null && lightingCallback != null) {
                    // Perform lighting analysis
                    LightingAnalyzer.LightingAnalysisResult result =
                            LightingAnalyzer.analyzeLighting(bitmap);

                    // Post result to main thread
                    ContextCompat.getMainExecutor(context).execute(() -> {
                        lightingCallback.onLightingAnalyzed(result);
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Error in lighting analysis", e);
            } finally {
                image.close();
            }
        }
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
     * Shutdown camera
     */
    public void shutdown() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        cameraExecutor.shutdown();
    }
}