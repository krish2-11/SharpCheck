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
import com.example.blurdetectionapp.utils.ShadowDetection;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import java.util.Objects;

@ExperimentalGetImage
public class MainActivity extends AppCompatActivity implements
        CameraManager.LightingAnalysisCallback, CameraManager.ImageCaptureCallback, CameraManager.BlurAnalysisCallback {
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_CODE = 200;

    // UI Components
    private PreviewView previewView;
    private TextView lightingStatusText;
    private TextView lightingDetailText;
    private Button captureButton;
    private Button toggleResultsButton;
    private ImageView imageView;
    private ImageView imageView2;
    private TextView resultText;
    private View resultsPanel;
    private OverlayView overlayView;
    private DocumentDetection documentDetection;

    private TextView blurStatusText;

    // Camera and Analysis
    private CameraManager cameraManager;
    private Handler mainHandler;

    // Current lighting analysis result
    private LightingAnalyzer.LightingAnalysisResult currentLightingResult;
    private BlurDetector.BlurDetectionResult currentBlurResult;

    private final Handler cornerHandler = new Handler(Looper.getMainLooper());
    private Runnable cornerRunnable;
    private boolean isCornerDetectionActive = false;
    private Bitmap latestFrame;


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
        blurStatusText = findViewById(R.id.BlurStatusText);

        // Control buttons
        captureButton = findViewById(R.id.captureButton);
        toggleResultsButton = findViewById(R.id.toggleResultsButton);
        Button backToCameraButton = findViewById(R.id.backToCameraButton);

        // Result views
        resultsPanel = findViewById(R.id.resultsPanel);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        resultText = findViewById(R.id.resultText);

        overlayView = findViewById(R.id.overlayView);

        // Set click listeners
        captureButton.setOnClickListener(v -> onCaptureClicked());
        toggleResultsButton.setOnClickListener(v -> toggleResultsView());
        backToCameraButton.setOnClickListener(v -> backToCameraView());

        documentDetection = new DocumentDetection();

        // Initially disable capture button until lighting analysis is done
        updateCaptureButtonState(false);
    }

    private void initializeCamera() {
        cameraManager = new CameraManager(this, this);
        cameraManager.initializeCamera(previewView, this, this, this);
        cameraManager.setFrameAnalyzerCallback(this::processFrame);
        startCornerDetectionLoop();
        Log.d(TAG, "Camera initialization started");
    }

    private void processFrame(Bitmap bitmap) {
        if (bitmap != null) {
            latestFrame = bitmap.copy(Objects.requireNonNull(bitmap.getConfig()), false); // keep a copy
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
//                    new Thread(() -> {
//                        Bitmap shadowMask = ShadowDetection.detectShadows(latestFrame);
//                        runOnUiThread(() -> overlayView.setShadowMask(shadowMask));
//                    }).start();
                }

                if (isCornerDetectionActive) {
                    cornerHandler.postDelayed(this, 2000); // repeat after 2 sec
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

        // 🔹 FIT_CENTER → use the smaller scale
        float scale = Math.min(scaleX, scaleY);

        // Add black borders (letterboxing/pillar boxing) padding
        float dx = (viewWidth - imgWidth * scale) / 2f;
        float dy = (viewHeight - imgHeight * scale) / 2f;

        PointF[] mapped = new PointF[4];
        for (int i = 0; i < 4; i++) {
            mapped[i] = new PointF(
                    (float) (points[i].x * scale + dx),
                    (float) (points[i].y * scale + dy)
            );
        }
        mapped = expandDocumentCorners(mapped , 1.3f);
        return mapped;
    }

    private PointF[] expandDocumentCorners(PointF[] corners, float expansionFactor) {
        // Compute center of the rectangle
        float centerX = 0, centerY = 0;
        for (PointF p : corners) {
            centerX += p.x;
            centerY += p.y;
        }
        centerX /= corners.length;
        centerY /= corners.length;

        // Move each point away from center by expansionFactor
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
            // Show dialog explaining why capture is disabled
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

    @SuppressLint("SetTextI18n")
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
            updateCaptureButtonState(result.isCaptureEnabled
            );
        });
    }

    private String generateLightingIssueExplanation(LightingAnalyzer.LightingAnalysisResult result) {
        StringBuilder explanation = new StringBuilder();

        if (result.hasReflection) {
            explanation.append("• Reflection detected on document surface\n");
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
            String blurStatus = "Image is " + blurResult.description;
            resultText.setText(blurStatus);

            // Detect document corners
            //Point[] corners
            Point[] corners = documentDetection.detectDocumentCornersPoints(bitmap);
            if (corners != null) {
                // Warp the document from detected corners
                Bitmap warpedBitmap = documentDetection.warpToDocumentFromPoints(bitmap, corners);
                // Show cropped document in imageView2
                imageView2.setImageBitmap(warpedBitmap);
            } else {
                Toast.makeText(this, "No document detected", Toast.LENGTH_SHORT).show();
            }

            // Show results panel
            showResultsView();

            // Re-enable capture button based on current lighting condition
            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled
                );
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
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled
                );
            } else {
                updateCaptureButtonState(false);
            }
        });
    }

    @SuppressLint("SetTextI18n")
    private void updateCaptureButtonState(boolean enabled) {
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

    @SuppressLint("SetTextI18n")
    private void showResultsView() {
        previewView.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.VISIBLE);
        toggleResultsButton.setVisibility(View.VISIBLE);
        toggleResultsButton.setText("Back to Camera");
    }

    @SuppressLint("SetTextI18n")
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
            isCornerDetectionActive = false;
            cornerHandler.removeCallbacks(cornerRunnable);
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

    @Override
    public void onBlurAnalyzed(BlurDetector.BlurDetectionResult result) {
        currentBlurResult = result;
        mainHandler.post(() -> {
            @SuppressLint("DefaultLocale") String blurMessage = String.format("Blur: %s (Variance: %.1f)",
                    result.description, result.laplacianVariance);
            blurStatusText.setText(blurMessage);
            // Disable capture button if image is blurred or lighting is bad
            boolean canCapture = (currentLightingResult != null && currentLightingResult.isCaptureEnabled)
                    && !result.isBlurred;
            updateCaptureButtonState(canCapture
            );
        });
    }

}