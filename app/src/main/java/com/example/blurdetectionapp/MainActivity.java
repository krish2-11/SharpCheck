package com.example.blurdetectionapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.blurdetectionapp.camera.CameraManager;
import com.example.blurdetectionapp.utils.BlurDetector;
import com.example.blurdetectionapp.utils.DocumentDetection;
import com.example.blurdetectionapp.utils.LightingAnalyzer;
import com.example.blurdetectionapp.utils.OverlayView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.Objects;

@ExperimentalGetImage
public class MainActivity extends AppCompatActivity implements
        CameraManager.LightingAnalysisCallback,
        CameraManager.ImageCaptureCallback,
        CameraManager.BlurAnalysisCallback {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_CODE = 200;

    // UI Components
    private PreviewView previewView;
    private TextView lightingStatusText;
    private TextView lightingDetailText;
    private TextView blurStatusText;
    private Button captureButton;
    private Button toggleResultsButton;
    private ImageView imageView;
    private ImageView imageView2;
    private TextView resultText;
    private View resultsPanel;
    private OverlayView overlayView;

    private DocumentDetection documentDetection;

    // Camera and Analysis
    private CameraManager cameraManager;
    private Handler mainHandler;

    // Current analysis results
    private LightingAnalyzer.LightingAnalysisResult currentLightingResult;
    private BlurDetector.BlurDetectionResult currentBlurResult;

    // Document detection loop
    private final Handler cornerHandler = new Handler(Looper.getMainLooper());
    private Runnable cornerRunnable;
    private boolean isCornerDetectionActive = false;
    private Bitmap latestFrame;

    // Captured image
    private Bitmap capturedBitmap;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed!");
        } else {
            Log.d(TAG, "OpenCV initialized successfully");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        mainHandler = new Handler(Looper.getMainLooper());

        // Check camera permission
        if (hasCameraPermission()) {
            initializeCamera();
        } else {
            requestCameraPermission();
        }
    }

    private void initializeViews() {
        previewView = findViewById(R.id.previewView);
        lightingStatusText = findViewById(R.id.lightingStatusText);
        lightingDetailText = findViewById(R.id.lightingDetailText);
        blurStatusText = findViewById(R.id.BlurStatusText);

        captureButton = findViewById(R.id.captureButton);
        toggleResultsButton = findViewById(R.id.toggleResultsButton);
        Button backToCameraButton = findViewById(R.id.backToCameraButton);

        resultsPanel = findViewById(R.id.resultsPanel);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        resultText = findViewById(R.id.resultText);

        overlayView = findViewById(R.id.overlayView);

        captureButton.setOnClickListener(v -> onCaptureClicked());
        toggleResultsButton.setOnClickListener(v -> toggleResultsView());
        backToCameraButton.setOnClickListener(v -> backToCameraView());

        documentDetection = new DocumentDetection();

        updateCaptureButtonState(false);
    }

    private void initializeCamera() {
        cameraManager = new CameraManager(this, this);
        cameraManager.initializeCamera(previewView, this, this, this);
        cameraManager.setFrameAnalyzerCallback(this::processFrame);
        startCornerDetectionLoop();
        Log.d(TAG, "Camera initialized");
    }

    private void processFrame(Bitmap bitmap) {
        if (bitmap != null) {
            latestFrame = bitmap.copy(Objects.requireNonNull(bitmap.getConfig()), false);
        }
    }

    private void startCornerDetectionLoop() {
        isCornerDetectionActive = true;
        cornerRunnable = new Runnable() {
            @Override
            public void run() {
                if (latestFrame != null) {
                    Point[] corners = documentDetection.detectDocumentCornersPoints(latestFrame);
                    if (corners != null) {
                        PointF[] mappedPoints = mapPointsToOverlay(
                                corners,
                                latestFrame.getWidth(),
                                latestFrame.getHeight(),
                                overlayView
                        );

                        runOnUiThread(() -> overlayView.setDocumentCorners(mappedPoints));
                    } else {
                        runOnUiThread(() -> overlayView.clearCorners());
                    }
                }

                if (isCornerDetectionActive) {
                    cornerHandler.postDelayed(this, 2000); // every 2s
                }
            }
        };
        cornerHandler.post(cornerRunnable);
    }

    private PointF[] mapPointsToOverlay(Point[] points, int imgWidth, int imgHeight, View overlayView) {
        int viewWidth = overlayView.getWidth();
        int viewHeight = overlayView.getHeight();

        if (viewWidth == 0 || viewHeight == 0) {
            PointF[] fallback = new PointF[4];
            for (int i = 0; i < 4; i++) {
                fallback[i] = new PointF((float) points[i].x, (float) points[i].y);
            }
            return fallback;
        }

        float scaleX = (float) viewWidth / imgWidth;
        float scaleY = (float) viewHeight / imgHeight;
        float scale = Math.min(scaleX, scaleY);

        float dx = (viewWidth - imgWidth * scale) / 2f;
        float dy = (viewHeight - imgHeight * scale) / 2f;

        PointF[] mapped = new PointF[4];
        for (int i = 0; i < 4; i++) {
            mapped[i] = new PointF(
                    (float) (points[i].x * scale + dx),
                    (float) (points[i].y * scale + dy)
            );
        }
        return expandDocumentCorners(mapped, 1.3f);
    }

    private PointF[] expandDocumentCorners(PointF[] corners, float expansionFactor) {
        float centerX = 0, centerY = 0;
        for (PointF p : corners) {
            centerX += p.x;
            centerY += p.y;
        }
        centerX /= corners.length;
        centerY /= corners.length;

        PointF[] expanded = new PointF[corners.length];
        for (int i = 0; i < corners.length; i++) {
            float dx = corners[i].x - centerX;
            float dy = corners[i].y - centerY;
            expanded[i] = new PointF(
                    centerX + dx * expansionFactor,
                    centerY + dy * expansionFactor
            );
        }
        return expanded;
    }

    @SuppressLint("SetTextI18n")
    private void onCaptureClicked() {
        if (currentLightingResult == null) {
            Toast.makeText(this, "Lighting analysis not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        if (currentLightingResult.lightingCondition == LightingAnalyzer.LightingCondition.BAD) {
            showLightingIssueDialog();
            return;
        }

        if (currentBlurResult != null && currentBlurResult.isBlurred) {
            Toast.makeText(this, "Image is too blurry. Please adjust focus.", Toast.LENGTH_SHORT).show();
            return;
        }

        if (cameraManager != null) {
            captureButton.setEnabled(false);
            captureButton.setText("Capturing...");
            cameraManager.captureImage();
        }
    }

    private void showLightingIssueDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Image Quality Issue")
                .setMessage("Cannot capture image due to lighting:\n\n" +
                        generateLightingIssueExplanation(currentLightingResult) +
                        "\n\nPlease adjust lighting or camera position.")
                .setPositiveButton("OK", (dialog, which) -> dialog.dismiss())
                .setNegativeButton("Capture Anyway", (dialog, which) -> {
                    if (cameraManager != null) {
                        captureButton.setEnabled(false);
                        captureButton.setText("Capturing...");
                        cameraManager.captureImage();
                    }
                    dialog.dismiss();
                })
                .show();
    }

    @Override
    public void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result) {
        currentLightingResult = result;
        mainHandler.post(() -> {
            lightingStatusText.setText(result.statusMessage);
            lightingDetailText.setText(result.detailMessage);

            boolean canCapture = result.isCaptureEnabled
                    && (currentBlurResult == null || !currentBlurResult.isBlurred);
            updateCaptureButtonState(canCapture);
        });
    }

    private String generateLightingIssueExplanation(LightingAnalyzer.LightingAnalysisResult result) {
        StringBuilder explanation = new StringBuilder();

        if (result.hasReflection) {
            explanation.append("• Reflection detected on document surface\n");
        }
        if (result.brightPixelRatio >= 0.55) {
            explanation.append("• Excessive brightness: possible overexposure\n");
        }
//        if (result.laplacianVariance <= 900) {
//            explanation.append("• Low contrast: insufficient detail\n");
//        }

        if (explanation.length() == 0) {
            explanation.append("Multiple lighting issues detected");
        }
        return explanation.toString();
    }

    @Override
    public void onBlurAnalyzed(BlurDetector.BlurDetectionResult result) {
        currentBlurResult = result;
        mainHandler.post(() -> {
            @SuppressLint("DefaultLocale")
            String blurMessage = String.format("Blur: %b, Occluded: %b, AvgVariance: %.1f, BlurPct: %.3f, OcclusionPct: %.3f)",
                    result.isBlurred, result.isOccluded,  result.avgVariance, result.blurPercentage, result.occlusionPercentage);

            blurStatusText.setText(blurMessage);

            boolean canCapture = (currentLightingResult != null && currentLightingResult.isCaptureEnabled)
                    && !result.isBlurred;
            updateCaptureButtonState(canCapture);
        });
    }

    @Override
    public void onImageCaptured(Bitmap bitmap) {
        capturedBitmap = bitmap;

        mainHandler.post(() -> {
            imageView.setImageBitmap(bitmap);

            // Document detection (unchanged)
            Point[] corners = documentDetection.detectDocumentCornersPoints(bitmap);
            if (corners != null) {
                Bitmap warpedBitmap = documentDetection.warpToDocumentFromPoints(bitmap, corners);
                imageView2.setImageBitmap(warpedBitmap);
            } else {
                Toast.makeText(this, "No document detected", Toast.LENGTH_SHORT).show();
            }

            showResultsView();

            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled);
            } else {
                updateCaptureButtonState(false);
            }
        });
    }

    @Override
    public void onCaptureError(String error) {
        mainHandler.post(() -> {
            Toast.makeText(this, "Capture failed: " + error, Toast.LENGTH_SHORT).show();
            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled);
            } else {
                updateCaptureButtonState(false);
            }
        });
    }

    @SuppressLint("SetTextI18n")
    private void updateCaptureButtonState(boolean enabled) {
        captureButton.setEnabled(enabled);
        captureButton.setText("CAPTURE");
    }

    private void toggleResultsView() {
        if (resultsPanel.getVisibility() == View.VISIBLE) {
            backToCameraView();
        } else {
            showResultsView();
        }
    }

    private void showResultsView() {
        previewView.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.VISIBLE);
        toggleResultsButton.setVisibility(View.VISIBLE);
        toggleResultsButton.setText("Back to Camera");
    }

    private void backToCameraView() {
        resultsPanel.setVisibility(View.GONE);
        previewView.setVisibility(View.VISIBLE);
        toggleResultsButton.setVisibility(capturedBitmap != null ? View.VISIBLE : View.GONE);
        toggleResultsButton.setText("Show Results");
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeCamera();
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        isCornerDetectionActive = false;
        cornerHandler.removeCallbacks(cornerRunnable);
        if (cameraManager != null) {
            cameraManager.shutdown();
        }
    }
}
