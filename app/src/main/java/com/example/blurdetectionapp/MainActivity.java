package com.example.blurdetectionapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
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
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.blurdetectionapp.camera.CameraManager;
import com.example.blurdetectionapp.utils.BlurDetector;
import com.example.blurdetectionapp.utils.DocumentDetection;
import com.example.blurdetectionapp.utils.ImageUtils;
import com.example.blurdetectionapp.utils.LightingAnalyzer;
import com.example.blurdetectionapp.utils.OverlayView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import java.util.List;
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
    private Button captureButton;
    private Button toggleResultsButton;
    private ImageView imageView;
    private ImageView imageView2;
    private View resultsPanel;
    private OverlayView overlayView;
    private TextView lightingBlurStatusText;
    private TextView lightingBlurDetailText;
    private ImageView star1, star2, star3;

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
    RectF overlayRect = null;

    private Point[] lastStableCorners = null;
    private android.util.Size lastFrameSize = null;

    private TextView modeRectangle, modeSquare, modeCard;

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

        captureButton = findViewById(R.id.captureButton);
        toggleResultsButton = findViewById(R.id.toggleResultsButton);
        Button backToCameraButton = findViewById(R.id.backToCameraButton);

        resultsPanel = findViewById(R.id.resultsPanel);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);

        lightingBlurStatusText = findViewById(R.id.lightingBlurStatusText);
        lightingBlurDetailText = findViewById(R.id.lightingBlurDetailText);
        star1 = findViewById(R.id.star1);
        star2 = findViewById(R.id.star2);
        star3 = findViewById(R.id.star3);


        modeRectangle=findViewById(R.id.modeRectangle);
        modeSquare=findViewById(R.id.modeSquare);
        modeCard=findViewById(R.id.modeCard);

        View.OnClickListener modeClickListener = v -> {
            String mode = ((TextView)v).getText().toString();
            onModeChanged(mode);
            highlightSelected((TextView)v);
        };

        modeSquare.setOnClickListener(modeClickListener);
        modeRectangle.setOnClickListener(modeClickListener);
        modeCard.setOnClickListener(modeClickListener);

        overlayView = findViewById(R.id.overlayView);

        onModeChanged("A4");
        highlightSelected(modeRectangle);

        captureButton.setOnClickListener(v -> onCaptureClicked());
        toggleResultsButton.setOnClickListener(v -> toggleResultsView());
        backToCameraButton.setOnClickListener(v -> backToCameraView());
        documentDetection = new DocumentDetection(this);

        updateCaptureButtonState(true);
    }

    private void onModeChanged(String mode) {
        OverlayView.OverlayType type = null;
        switch (mode) {
            case "A4":
                type = OverlayView.OverlayType.PORTRAIT;
                break;
            case "Square":
                type = OverlayView.OverlayType.SQUARE;
                break;
            case "Card":
                type = OverlayView.OverlayType.LANDSCAPE;
        }

        // Set overlay without bitmap, just the type for proper sizing
        if (type != null) {
            overlayView.setOverlay(null, type);
            overlayRect = overlayView.calculateOverlayRect();
        }
    }

    private void highlightSelected(TextView selected) {
        // Reset all
        modeCard.setTextColor(Color.WHITE);
        modeRectangle.setTextColor(Color.WHITE);
        modeSquare.setTextColor(Color.WHITE);

        // Highlight current
        selected.setTextColor(Color.YELLOW);
    }

    private void initializeCamera() {
        cameraManager = new CameraManager(this, this);
        cameraManager.initializeCamera(previewView, this, this, this);
        setFrameAnalyzerCallback(this::processFrame);
        cameraManager.setOverlayView(overlayView);
        startCornerDetectionLoop();
        Log.d(TAG, "Camera initialized");
    }

    private void processFrame(Bitmap bitmap) {
        if (bitmap != null) {
            latestFrame = bitmap.copy(Objects.requireNonNull(bitmap.getConfig()), false);
        }
    }

    public void setFrameAnalyzerCallback(CameraManager.FrameCallback callback) {
        if (cameraManager != null) {
            cameraManager.setFrameCallback(callback);
        }
    }

    // NEW METHOD: Extract ROI bitmap from the overlay area
    private Bitmap extractROIFromFrame(Bitmap frame) {
        if (frame == null || overlayView == null) return frame;

        RectF overlayRect = overlayView.getOverlayRect();
        if (overlayRect == null) {
            Log.w(TAG, "OverlayRect is null during extractROIFromFrame, using full frame");
            return frame;
        }

        try {
            // Scale overlay rect from view coordinates to bitmap coordinates
            Rect bitmapRect = scaleRectToBitmap(overlayRect,
                    frame.getWidth(), frame.getHeight(),
                    overlayView.getWidth(), overlayView.getHeight());

            // Ensure rect is within bitmap bounds
            bitmapRect.left = Math.max(0, bitmapRect.left);
            bitmapRect.top = Math.max(0, bitmapRect.top);
            bitmapRect.right = Math.min(frame.getWidth(), bitmapRect.right);
            bitmapRect.bottom = Math.min(frame.getHeight(), bitmapRect.bottom);

            if (bitmapRect.width() <= 0 || bitmapRect.height() <= 0) {
                Log.w(TAG, "Invalid bitmap rect, using full frame");
                return frame;
            }

            return Bitmap.createBitmap(frame,
                    bitmapRect.left, bitmapRect.top,
                    bitmapRect.width(), bitmapRect.height());
        } catch (Exception e) {
            Log.e(TAG, "Error extracting ROI: " + e.getMessage());
            return frame; // fallback to full frame
        }
    }

    private void startCornerDetectionLoop() {
        isCornerDetectionActive = true;
        cornerRunnable = new Runnable() {
            @Override
            public void run() {
                if (latestFrame != null && !latestFrame.isRecycled()) {
                    try {
                        Bitmap roiBitmap = extractROIFromFrame(latestFrame);
                        if (roiBitmap != null) {
                            Bitmap rotatedFrame = ImageUtils.rotateBitmap(roiBitmap, 90);

                            // Detect corners in the full rotated frame
                            List<Point[]> allCorners = documentDetection.detectDocumentCornersPoints(rotatedFrame);

                            if (allCorners != null && !allCorners.isEmpty()) {
                                // The method already finds the best one, so just get the first
                                Point[] corners = allCorners.get(0);

                                lastStableCorners = corners;
                                lastFrameSize = new android.util.Size(rotatedFrame.getWidth(), rotatedFrame.getHeight());

                                // --- FIX 3: Map corners from the full frame's coordinates to the view's coordinates ---
                                PointF[] mappedPoints = mapPointsToOverlay(corners, rotatedFrame.getWidth(), rotatedFrame.getHeight(), overlayView);
                                runOnUiThread(() -> overlayView.setDocumentCorners(mappedPoints));
                            } else {
                                runOnUiThread(() -> overlayView.clearCorners());
                            }

                            // Clean up the rotated bitmap
                            if (rotatedFrame != latestFrame) {
                                rotatedFrame.recycle();
                            }
                        }

                    } catch (Exception e) {
                        Log.e(TAG, "Error in corner detection loop: " + e.getMessage());
                        runOnUiThread(() -> overlayView.clearCorners());
                    }
                }

                if (isCornerDetectionActive) {
                    // Run more frequently for a smoother feel
                    cornerHandler.postDelayed(this, 100);
                }
            }
        };
        cornerHandler.post(cornerRunnable);
    }

    private PointF[] mapPointsToOverlay(Point[] points, int imgWidth, int imgHeight, View overlayView) {
        int viewWidth = overlayView.getWidth();
        int viewHeight = overlayView.getHeight();

        if (viewWidth == 0 || viewHeight == 0) return null; // Cannot map if view is not ready

        // This logic correctly handles fitting the image preview (which may be letterboxed) inside the view
        float scaleX = (float) viewWidth / imgWidth;
        float scaleY = (float) viewHeight / imgHeight;
        float scale = Math.min(scaleX, scaleY); // Use min to maintain aspect ratio

        // Calculates the offset for letterboxing/pillarboxing
        float dx = (viewWidth - imgWidth * scale) / 2f;
        float dy = (viewHeight - imgHeight * scale) / 2f;

        PointF[] mapped = new PointF[4];
        for (int i = 0; i < 4; i++) {
            mapped[i] = new PointF(
                    (float) (points[i].x * scale + dx),
                    (float) (points[i].y * scale + dy)
            );
        }
        // --- FIX 4: Do NOT artificially expand the corners ---
        return mapped;
    }

    @SuppressLint("SetTextI18n")
    private void onCaptureClicked() {
        if (cameraManager != null) {
            //captureButton.setEnabled(false);
            captureButton.setText("Capturing..");
            cameraManager.captureImage();
        }
    }

    @Override
    public void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result) {
        currentLightingResult = result;
        mainHandler.post(this::updateStarRatingAndStatus);
    }

    @Override
    public void onImageCaptured(Bitmap bitmap) {
        capturedBitmap = bitmap;

        mainHandler.post(() -> {

            if (lastStableCorners == null) {
                Toast.makeText(this, "No document detected in preview.", Toast.LENGTH_SHORT).show();
                imageView2.setImageResource(R.drawable.no_document);
                return;
            }

            Bitmap roiBitmap = extractROIFromFrame(latestFrame != null ? latestFrame : bitmap);
            if (roiBitmap != null) {
            roiBitmap = ImageUtils.rotateBitmap(roiBitmap , 90);
            imageView.setImageBitmap(roiBitmap);

            // 3. Detect corners in ROI (optional)
            List<Point[]> detectedCorners = documentDetection.detectDocumentCornersPoints(roiBitmap);
            Point[] roiCorners;
            if (detectedCorners != null && !detectedCorners.isEmpty()) {
                roiCorners = detectedCorners.get(0);
            } else {
                // fallback: use lastStableCorners relative to ROI
                roiCorners = new Point[lastStableCorners.length];
                for (int i = 0; i < lastStableCorners.length; i++) {
                    roiCorners[i] = new Point(
                            lastStableCorners[i].x,
                            lastStableCorners[i].y
                    );
                }
            }

            // 4. Warp ROI
            Bitmap warpedBitmap = documentDetection.warpToDocumentFromPoints(roiBitmap, roiCorners);
            // 5. Display
            imageView2.setImageBitmap(warpedBitmap);
            //roiBitmap2.recycle();
            }

            showResultsView();
            updateCaptureButtonState(true);
        });
    }

    private Rect scaleRectToBitmap(RectF rectInView, int bmpW, int bmpH, int viewW, int viewH) {
        if (viewW <= 0 || viewH <= 0) {
            Log.w(TAG, "Invalid view dimensions (w=" + viewW + ", h=" + viewH + "). Falling back to full bitmap.");
            return new Rect(0, 0, bmpW, bmpH);  // Full frame fallback
        }
        if (bmpW <= 0 || bmpH <= 0) {
            Log.w(TAG, "Invalid bitmap dimensions (w=" + bmpW + ", h=" + bmpH + "). Returning empty rect.");
            return new Rect(0, 0, 0, 0);  // Invalid, will trigger full fallback in caller
        }

        float scaleX = (float) viewW / bmpW;
        float scaleY = (float) viewH / bmpH;
        float scale = Math.min(scaleX, scaleY);

        float dx = (viewW - bmpW * scale) / 2f;
        float dy = (viewH - bmpH * scale) / 2f;

        int left   = Math.round((rectInView.left   - dx) / scale);
        int top    = Math.round((rectInView.top    - dy) / scale);
        int right  = Math.round((rectInView.right  - dx) / scale);
        int bottom = Math.round((rectInView.bottom - dy) / scale);

        // Clamp to bitmap bounds
        left   = Math.max(0, Math.min(left,   bmpW - 1));
        top    = Math.max(0, Math.min(top,    bmpH - 1));
        right  = Math.max(0, Math.min(right,  bmpW));
        bottom = Math.max(0, Math.min(bottom, bmpH));

        return new Rect(left, top, right, bottom);
    }

    @Override
    public void onCaptureError(String error) {
        mainHandler.post(() -> {
            Toast.makeText(this, "Capture failed: " + error, Toast.LENGTH_SHORT).show();
//            if (currentLightingResult != null) {
//                updateCaptureButtonState(currentLightingResult.isCaptureEnabled);
//            } else {
//                //updateCaptureButtonState(false);
//            }
//            updateCaptureButtonState(true);
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
        mainHandler.post(this::updateStarRatingAndStatus);
    }

    @SuppressLint("SetTextI18n")
    private void updateStarRatingAndStatus() {
        if (currentLightingResult == null || currentBlurResult == null) {
            lightingBlurStatusText.setText("Analyzing...");
            lightingBlurDetailText.setText("");
            setStars(0);
            updateCaptureButtonState(false);
            return;
        }

        int stars = 3; // max stars

        // Deduct stars for bad lighting
        if (currentLightingResult.lightingCondition == LightingAnalyzer.LightingCondition.BAD) {
            stars = Math.min(stars, 1);
        } else if (currentLightingResult.lightingCondition == LightingAnalyzer.LightingCondition.GOOD) {
            stars = Math.min(stars, 2);
        }

        // Deduct stars for blur
        if (currentBlurResult.isBlurred) {
            stars = Math.min(stars, 1);
        } else if (currentBlurResult.blurPercentage > 0.1) { // example threshold
            stars = Math.min(stars, 2);
        }
        // Update star images
        setStars(stars);

        // Update status text
        String statusText;
        switch (stars) {
            case 3:
                statusText = "Excellent Image Quality";
                break;
            case 2:
                statusText = "Good Image Quality";
                break;
            case 1:
                statusText = "Poor Image Quality";
                break;
            default:
                statusText = "Analyzing...";
        }
        lightingBlurStatusText.setText(statusText);

        // Optionally show details from lighting and blur
        String detail = "";
        if (currentLightingResult.hasReflection) {
            detail += "Reflection detected. ";
        }
        if (currentLightingResult.brightPixelRatio >= 0.55) {
            detail += "Overexposed. ";
        }
        if (currentBlurResult.isBlurred) {
            detail += "Image is blurry. ";
        }
        lightingBlurDetailText.setText(detail.trim());

//         Enable capture only if stars >= 2
        updateCaptureButtonState(stars >= 2);
    }

    private void setStars(int count) {
        star1.setImageResource(count >= 1 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
        star2.setImageResource(count >= 2 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
        star3.setImageResource(count >= 3 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
    }
}