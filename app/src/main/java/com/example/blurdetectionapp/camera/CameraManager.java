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

import com.example.blurdetectionapp.utils.BlurDetector;
import com.example.blurdetectionapp.utils.LightingAnalyzer;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Optimized Camera manager using CameraX + OpenCV
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

    // Callbacks
    private LightingAnalysisCallback lightingCallback;
    private BlurAnalysisCallback blurCallback;
    private ImageCaptureCallback captureCallback;

    public interface LightingAnalysisCallback {
        void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result);
    }

    public interface FrameCallback {
        void onFrameAvailable(Bitmap bitmap);
    }

    private FrameCallback frameCallback;

    public void setFrameAnalyzerCallback(FrameCallback callback) {
        this.frameCallback = callback;
    }


    public interface BlurAnalysisCallback {
        void onBlurAnalyzed(BlurDetector.BlurDetectionResult result);
    }

    public interface ImageCaptureCallback {
        void onImageCaptured(Bitmap bitmap);
        void onCaptureError(String error);
    }

    public CameraManager(Context context, LifecycleOwner lifecycleOwner) {
        this.context = context;
        this.lifecycleOwner = lifecycleOwner;
        this.cameraExecutor = Executors.newFixedThreadPool(2); // allow parallel analysis
    }

    /** Initialize camera */
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

    private void startCamera(PreviewView previewView) {
        preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build();

        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480)) // low res for analysis
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, new CombinedImageAnalyzer());

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        try {
            cameraProvider.unbindAll();
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

    /** Combined Analyzer using OpenCV */
    private class CombinedImageAnalyzer implements ImageAnalysis.Analyzer {
        private long lastLightingCheck = 0;
        private static final long LIGHTING_ANALYSIS_INTERVAL = 500; // ms
        private long lastBlurCheck = 0;
        private static final long BLUR_ANALYSIS_INTERVAL = 300; // ms
        @Override
        public void analyze(@NonNull ImageProxy image) {
            long now = System.currentTimeMillis();
            Mat mat = imageProxyToMat(image); // 🔑 convert directly to Mat
            image.close();
            if (mat == null) return;

//            // ✅ Throttled Lighting Analysis
//            if (lightingCallback != null && now - lastLightingCheck >= LIGHTING_ANALYSIS_INTERVAL) {
//                lastLightingCheck = now;
//                Mat lightingMat = mat.clone();
//                cameraExecutor.execute(() -> {
//                    LightingAnalyzer.LightingAnalysisResult result =
//                            LightingAnalyzer.analyzeLighting(lightingMat);
//                    lightingMat.release();
//                    ContextCompat.getMainExecutor(context).execute(() ->
//                            lightingCallback.onLightingAnalyzed(result)
//                    );
//                });
//            }

            // ✅ Throttled Blur Analysis
            if (blurCallback != null && now - lastBlurCheck >= BLUR_ANALYSIS_INTERVAL) {
                lastBlurCheck = now;
                Mat blurMat = mat.clone();
                cameraExecutor.execute(() -> {
                    BlurDetector.BlurDetectionResult result =
                            BlurDetector.detectBlurAndOcclusion(blurMat);
                    blurMat.release();
                    ContextCompat.getMainExecutor(context).execute(() ->
                            blurCallback.onBlurAnalyzed(result)
                    );
                });
            }

            // ✅ Frame Callback (if needed for overlays)
            if (frameCallback != null) {
                Bitmap bitmap = matToBitmap(mat); // only when UI needs it
                ContextCompat.getMainExecutor(context).execute(() ->
                        frameCallback.onFrameAvailable(bitmap)
                );
            }

            mat.release();
        }

    }

    private Bitmap matToBitmap(Mat mat) {
        Bitmap bitmap;
        // Create a Bitmap with the same size as the Mat
        bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);

        // Convert Mat → Bitmap
        Utils.matToBitmap(mat, bitmap);

        return bitmap;
    }



    /** Capture image to bitmap */
    public void captureImage() {
        if (imageCapture == null) {
            if (captureCallback != null) captureCallback.onCaptureError("Camera not initialized");
            return;
        }

        imageCapture.takePicture(ContextCompat.getMainExecutor(context),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        Bitmap bitmap = imageProxyToBitmap(image);
                        image.close();
                        if (bitmap != null && captureCallback != null) {
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
                });
    }

    /** Convert ImageProxy to OpenCV Mat */
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

    /** For capture only: convert ImageProxy to Bitmap */
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        Image mediaImage = image.getImage();
        if (mediaImage == null) return null;
        try {
            ByteBuffer buffer = mediaImage.getPlanes()[0].getBuffer();
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e);
            return null;
        }
    }

    public void shutdown() {
        if (cameraProvider != null) cameraProvider.unbindAll();
        cameraExecutor.shutdownNow();
    }
}
