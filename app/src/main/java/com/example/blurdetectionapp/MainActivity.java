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
                        // --- FIX 1: Use the full frame, do NOT extract ROI ---
                        // --- FIX 2: Rotate by 90 degrees, NOT 180 ---
                        Bitmap rotatedFrame = ImageUtils.rotateBitmap(latestFrame, 90);

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

    // NEW METHOD: Map corners from ROI coordinates to overlay view coordinates
    private PointF[] mapROICornersToOverlay(Point[] corners, Bitmap roiBitmap) {
        if (overlayRect == null) {
            return mapPointsToOverlay(corners, roiBitmap.getWidth(), roiBitmap.getHeight(), overlayView);
        }

        // Map corners from ROI bitmap coordinates to overlay view coordinates
        PointF[] mapped = new PointF[4];
        for (int i = 0; i < 4; i++) {
            float x = (float) (corners[i].x * overlayRect.width() / roiBitmap.getWidth());
            float y = (float) (corners[i].y * overlayRect.height() / roiBitmap.getHeight());
            mapped[i] = new PointF(overlayRect.left + x, overlayRect.top + y);
        }

        return mapped;
    }

    private Point[] getLargestDocumentCorners(List<Point[]> allCorners, DocumentDetection detector) {
        if (allCorners == null || allCorners.isEmpty()) {
            return null;
        }
        // allCorners is already sorted by area descending in DocumentDetection, so take first
        return allCorners.get(0);
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

    private PointF[] expandDocumentCorners(PointF[] corners) {
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
                    centerX + dx * (float) 1.25,
                    centerY + dy * (float) 1.25
            );
        }
        return expanded;
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
            if (lastStableCorners == null || lastFrameSize == null) {
                Toast.makeText(this, "No document was detected to capture.", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                imageView2.setImageBitmap(bitmap);
                showResultsView();
                updateCaptureButtonState(true);
                return;
            }

            Log.d(TAG, "Using last stable corners for capture: " + java.util.Arrays.toString(lastStableCorners));
            imageView.setImageBitmap(bitmap);

            // --- FINAL, MATHEMATICALLY CORRECT SCALING AND WARPING LOGIC ---

            // 1. Get dimensions
            // High-resolution original capture (e.g., 3000x4000)
            float highResOriginalWidth = bitmap.getWidth();
            float highResOriginalHeight = bitmap.getHeight();
            // Low-resolution rotated preview (e.g., 640x480)
            float lowResRotatedWidth = lastFrameSize.getWidth();
            float lowResRotatedHeight = lastFrameSize.getHeight();

            // 2. Calculate separate scale factors for X and Y to handle all aspect ratios.
            float scaleX = highResOriginalWidth / lowResRotatedHeight;
            float scaleY = highResOriginalHeight / lowResRotatedWidth;

            Point[] originalCorners = new Point[4];
            for (int i = 0; i < 4; i++) {
                Point lowResCorner = lastStableCorners[i];

                // 3. This is the corrected transformation from a +90 degree rotated space.
                // The original X coordinate comes from the rotated Y coordinate.
                // The original Y coordinate comes from the rotated X coordinate.
                double original_x = lowResCorner.y * scaleX;
                double original_y = (lowResRotatedWidth - lowResCorner.x) * scaleY;

                originalCorners[i] = new Point(original_x, original_y);
            }

            // 4. Warp the ORIGINAL high-resolution image using the final, perfectly mapped corners.
            Bitmap warpedBitmap = documentDetection.warpToDocumentFromPoints(bitmap, originalCorners);

            if (warpedBitmap != null) {
                imageView2.setImageBitmap(warpedBitmap);
            } else {
                Toast.makeText(this, "Could not create a top-down view.", Toast.LENGTH_SHORT).show();
                imageView2.setImageResource(R.drawable.no_document);
            }

            showResultsView();
            updateCaptureButtonState(true);
        });
    }

    // NEW METHOD: Scale corners from ROI to captured image coordinates
    private Point[] scaleCornersToCapturedImage(Point[] roiCorners, Bitmap roiBitmap, Bitmap capturedBitmap) {
        if (roiCorners == null || capturedBitmap == null) {
            Log.w(TAG, "Invalid inputs for corner scaling; returning null.");
            return null;
        }
        if (overlayRect == null) {
            Log.w(TAG, "overlayRect null; falling back to full image corners (no scaling).");
            // Detect corners directly on full captured (add this fallback)
            Bitmap rotatedCaptured = ImageUtils.rotateBitmap(capturedBitmap, 180);
            List<Point[]> allCorners = documentDetection.detectDocumentCornersPoints(rotatedCaptured);
            Point[] fullCorners = getLargestDocumentCorners(allCorners, documentDetection);
            if (rotatedCaptured != capturedBitmap) rotatedCaptured.recycle();
            return fullCorners != null ? shrinkBottomCorners(fullCorners) : null;  // Use full corners
        }

        Rect capturedROI = scaleRectToBitmap(overlayRect, capturedBitmap.getWidth(), capturedBitmap.getHeight(),
                overlayView.getWidth(), overlayView.getHeight());

        Point[] scaledCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            double x = roiCorners[i].x * capturedROI.width() / roiBitmap.getWidth();
            double y = roiCorners[i].y * capturedROI.height() / roiBitmap.getHeight();
            scaledCorners[i] = new Point(capturedROI.left + x, capturedROI.top + y);
        }
        return shrinkBottomCorners(scaledCorners);  // Now consistent
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


    private PointF[] shrinkBottomCorners(PointF[] corners) {
        if (corners == null || corners.length < 4) return corners;

        // Find min and max Y
        float minY = Float.MAX_VALUE;
        float maxY = Float.MIN_VALUE;
        for (PointF p : corners) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        float height = maxY - minY;
        if (height <= 0) return corners;

        float shrink = height * 0.05f; // e.g. 0.05 = shrink by 5%

        // Find two bottom-most points
        int idx1 = -1, idx2 = -1;
        float max1 = Float.MIN_VALUE, max2 = Float.MIN_VALUE;
        for (int i = 0; i < corners.length; i++) {
            if (corners[i].y > max1) {
                max2 = max1;
                idx2 = idx1;
                max1 = corners[i].y;
                idx1 = i;
            } else if (corners[i].y > max2) {
                max2 = corners[i].y;
                idx2 = i;
            }
        }

        // Move bottom two upward
        if (idx1 >= 0) corners[idx1].y = Math.max(corners[idx1].y - shrink, minY);
        if (idx2 >= 0) corners[idx2].y = Math.max(corners[idx2].y - shrink, minY);

        return corners;
    }

    private Point[] shrinkBottomCorners(Point[] corners) {
        if (corners == null || corners.length < 4) return corners;

        // Find min and max Y
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;
        for (Point p : corners) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        double height = maxY - minY;
        if (height <= 0) return corners;

        double shrink = height * 0.15; // e.g. 0.15 = shrink by 15%

        // Find two bottom-most points
        int idx1 = -1, idx2 = -1;
        double max1 = Double.MIN_VALUE, max2 = Double.MIN_VALUE;
        for (int i = 0; i < corners.length; i++) {
            if (corners[i].y > max1) {
                max2 = max1;
                idx2 = idx1;
                max1 = corners[i].y;
                idx1 = i;
            } else if (corners[i].y > max2) {
                max2 = corners[i].y;
                idx2 = i;
            }
        }

        // Move bottom two upward
        if (idx1 >= 0) corners[idx1].y = Math.max(corners[idx1].y - shrink, minY);
        if (idx2 >= 0) corners[idx2].y = Math.max(corners[idx2].y - shrink, minY);

        return corners;
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