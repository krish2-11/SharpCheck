package com.example.blurdetectionapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
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
import com.example.blurdetectionapp.utils.DocumentProcessor;
import com.example.blurdetectionapp.utils.LightingAnalyzer;

import org.opencv.android.OpenCVLoader;

@ExperimentalGetImage
public class MainActivity extends AppCompatActivity implements
        CameraManager.LightingAnalysisCallback, CameraManager.ImageCaptureCallback {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_CODE = 200;

    // UI Components
    private PreviewView previewView;
    private TextView lightingStatusText;
    private TextView lightingDetailText;
    private Button captureButton;
    private Button toggleResultsButton;
    private Button backToCameraButton;
    private ImageView imageView;
    private ImageView imageView2;
    private TextView resultText;
    private View resultsPanel;

    // Camera and Analysis
    private CameraManager cameraManager;
    private Handler mainHandler;

    // Current lighting analysis result
    private LightingAnalyzer.LightingAnalysisResult currentLightingResult;

    // Captured image data
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

        // Check camera permission and initialize
        if (hasCameraPermission()) {
            initializeCamera();
        } else {
            requestCameraPermission();
        }
    }

    private void initializeViews() {
        // Camera views
        previewView = findViewById(R.id.previewView);
        lightingStatusText = findViewById(R.id.lightingStatusText);
        lightingDetailText = findViewById(R.id.lightingDetailText);

        // Control buttons
        captureButton = findViewById(R.id.captureButton);
        toggleResultsButton = findViewById(R.id.toggleResultsButton);
        backToCameraButton = findViewById(R.id.backToCameraButton);

        // Result views
        resultsPanel = findViewById(R.id.resultsPanel);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        resultText = findViewById(R.id.resultText);

        // Set click listeners
        captureButton.setOnClickListener(v -> onCaptureClicked());
        toggleResultsButton.setOnClickListener(v -> toggleResultsView());
        backToCameraButton.setOnClickListener(v -> backToCameraView());

        // Initially disable capture button until lighting analysis is done
        updateCaptureButtonState(false, "Initializing camera...");
    }

    private void initializeCamera() {
        cameraManager = new CameraManager(this, this);
        cameraManager.initializeCamera(previewView, this, this);
        Log.d(TAG, "Camera initialization started");
    }

    private void onCaptureClicked() {
        if (currentLightingResult == null) {
            Toast.makeText(this, "Lighting analysis not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        if (currentLightingResult.lightingCondition == LightingAnalyzer.LightingCondition.BAD) {
            // Show dialog explaining why capture is disabled
            showLightingIssueDialog();
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
                .setMessage("Cannot capture image due to lighting conditions:\n\n" +
                        generateLightingIssueExplanation(currentLightingResult) +
                        "\n\nPlease adjust lighting or camera position for better image quality.")
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

    // CameraManager.LightingAnalysisCallback implementation
    @Override
    public void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result) {
        mainHandler.post(() -> {
            currentLightingResult = result;

            // Update UI
            lightingStatusText.setText(result.statusMessage);
            lightingDetailText.setText(result.detailMessage);

            // Update capture button state based on new LightingAnalyzer result
            updateCaptureButtonState(result.isCaptureEnabled,
                    result.isCaptureEnabled ? "Ready to capture" : "Poor lighting conditions");
        });
    }

    private String generateLightingIssueExplanation(LightingAnalyzer.LightingAnalysisResult result) {
        StringBuilder explanation = new StringBuilder();

        if (result.hasReflection) {
            explanation.append("• Reflection detected on document surface\n");
        }
        if (result.pixelSaturationRatio >= 0.12) {
            explanation.append("• High saturation: Too many overexposed or underexposed areas\n");
        }
        if (result.brightPixelRatio >= 0.55) {
            explanation.append("• Excessive brightness: Image may be overexposed\n");
        }
        if (result.laplacianVariance <= 900) {
            explanation.append("• Low contrast: Insufficient detail and edge definition\n");
        }

        if (explanation.length() == 0) {
            explanation.append("Multiple lighting quality indicators are suboptimal");
        }

        return explanation.toString();
    }

    // CameraManager.ImageCaptureCallback implementation
    @Override
    public void onImageCaptured(Bitmap bitmap) {
        capturedBitmap = bitmap;

        mainHandler.post(() -> {
            // Show captured image
            imageView.setImageBitmap(bitmap);

            // Perform blur detection
            BlurDetector.BlurDetectionResult blurResult = BlurDetector.detectBlur(bitmap);
            String blurStatus = blurResult.isBlurred ?
                    "Image is " + blurResult.description :
                    "Image is " + blurResult.description;
            resultText.setText(blurStatus);

            // Perform document detection
            DocumentProcessor.DocumentDetectionResult docResult =
                    DocumentProcessor.detectAndProcessDocument(bitmap);

            if (docResult.documentDetected && docResult.processedImage != null) {
                imageView2.setImageBitmap(docResult.processedImage);
            } else {
                imageView2.setImageBitmap(null);
                Toast.makeText(this, docResult.statusMessage, Toast.LENGTH_SHORT).show();
            }

            // Show results panel
            showResultsView();

            // Re-enable capture button based on current lighting condition
            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled,
                        currentLightingResult.isCaptureEnabled ? "Ready to capture" : "Poor lighting conditions");
            } else {
                updateCaptureButtonState(false, "Poor lighting conditions");
            }
        });
    }

    @Override
    public void onCaptureError(String error) {
        mainHandler.post(() -> {
            Toast.makeText(this, "Capture failed: " + error, Toast.LENGTH_SHORT).show();
            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled,
                        currentLightingResult.isCaptureEnabled ? "Ready to capture" : "Poor lighting conditions");
            } else {
                updateCaptureButtonState(false, "Poor lighting conditions");
            }
        });
    }

    private void updateCaptureButtonState(boolean enabled, String statusMessage) {
        captureButton.setEnabled(enabled);
        captureButton.setText("CAPTURE");

        // Optionally update lighting detail text with status message if needed
        // lightingDetailText.setText(statusMessage);
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

    // Permission handling
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
                Toast.makeText(this, "Camera permission required for this app", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraManager != null) {
            cameraManager.shutdown();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Camera will be paused automatically by lifecycle
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Camera will be resumed automatically by lifecycle
    }
}
