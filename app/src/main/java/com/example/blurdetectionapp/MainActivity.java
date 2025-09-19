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
import com.example.blurdetectionapp.utils.ShadowDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.util.Objects;

@ExperimentalGetImage
public class MainActivity extends AppCompatActivity implements
        CameraManager.LightingAnalysisCallback,
        CameraManager.ImageCaptureCallback,
        CameraManager.BlurAnalysisCallback,
        CameraManager.ShadowDetectionCallback{

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_CODE = 200;

    // UI Components
    private PreviewView previewView;
    private TextView lightingStatusText;
    private TextView lightingDetailText;

    private ImageView shadowMaskView;
    private TextView blurStatusText;
    private Button captureButton;
    private Button toggleResultsButton;
    private ImageView imageView;
    private ImageView imageView2;
    private TextView resultText;
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

    private ShadowDetector.ShadowDetectionResult currentShadowResult;

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
//        lightingStatusText = findViewById(R.id.lightingStatusText);
//        lightingDetailText = findViewById(R.id.lightingDetailText);
//        blurStatusText = findViewById(R.id.BlurStatusText);

        captureButton = findViewById(R.id.captureButton);
        toggleResultsButton = findViewById(R.id.toggleResultsButton);
        Button backToCameraButton = findViewById(R.id.backToCameraButton);

        resultsPanel = findViewById(R.id.resultsPanel);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
//        resultText = findViewById(R.id.resultText);

        lightingBlurStatusText = findViewById(R.id.lightingBlurStatusText);
        lightingBlurDetailText = findViewById(R.id.lightingBlurDetailText);
        star1 = findViewById(R.id.star1);
        star2 = findViewById(R.id.star2);
        star3 = findViewById(R.id.star3);

        shadowMaskView = findViewById(R.id.shadowMaskView);

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

        captureButton.setOnClickListener(v -> onCaptureClicked());
        toggleResultsButton.setOnClickListener(v -> toggleResultsView());
        backToCameraButton.setOnClickListener(v -> backToCameraView());

        captureButton.setEnabled(false);
//        captureButton.setEnabled(true);
        documentDetection = new DocumentDetection();

//        onModeChanged("Square");  // This sets the overlayView and overlayRect
//        highlightSelected(modeSquare);  // Highlight the Square mode button by default

        updateCaptureButtonState(false);
//        updateCaptureButtonState(true);
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

//        Toast.makeText(this, "Mode: " + mode, Toast.LENGTH_SHORT).show();
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
        cameraManager.setShadowDetectionCallback(this);
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
        if (frame == null || overlayView == null) return null;

        RectF overlayRect = overlayView.getOverlayRect();

        if (overlayRect == null){
            Log.e("OverlayRect" , "Its null during extractROIFromFrame");
            return frame;
        } // fallback to full frame

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
            return frame; // fallback to full frame
        }

        try {
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
                if (latestFrame != null) {
                    // Extract ROI before document detection
                    Bitmap roiBitmap = extractROIFromFrame(latestFrame);
                    if (roiBitmap != null) {
                        roiBitmap = ImageUtils.rotateBitmap(roiBitmap , 180);
                        //roiBitmap = ImageUtils.mirrorHorizontal(roiBitmap);
                        Point[] corners = documentDetection.detectDocumentCornersPoints(roiBitmap);
                        if (corners != null) {
                            // Map corners from ROI space back to overlay view coordinates
                            PointF[] mappedPoints = mapROICornersToOverlay(corners, roiBitmap);
                            runOnUiThread(() -> overlayView.setDocumentCorners(mappedPoints));
                        } else {
                            runOnUiThread(() -> overlayView.clearCorners());
                        }

                        // Clean up ROI bitmap if it's different from original
                        if (roiBitmap != latestFrame) {
                            roiBitmap.recycle();
                        }
                    }
                }

                if (isCornerDetectionActive) {
                    cornerHandler.postDelayed(this, 300);
                }
            }
        };
        cornerHandler.post(cornerRunnable);
    }

    // NEW METHOD: Map corners from ROI coordinates to overlay view coordinates
    private PointF[] mapROICornersToOverlay(Point[] corners, Bitmap roiBitmap) {
        if (overlayRect == null) {
            // Fallback to original mapping method
            return mapPointsToOverlay(corners, roiBitmap.getWidth(), roiBitmap.getHeight(), overlayView);
        }

        // 1) map to overlay co-ords
        PointF[] mapped = new PointF[4];
        for (int i = 0; i < 4; i++) {
            float x = (float) (corners[i].x * overlayRect.width() / roiBitmap.getWidth());
            float y = (float) (corners[i].y * overlayRect.height() / roiBitmap.getHeight());
            mapped[i] = new PointF(overlayRect.left + x, overlayRect.top + y);
        }

        // 2) expand first (keep existing behavior)
        mapped = expandDocumentCorners(mapped, 1.2f);

        // 3) compute top & bottom Y of expanded quad
        float minY = Float.MAX_VALUE;
        float maxY = Float.MIN_VALUE;
        for (PointF p : mapped) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }

        float height = maxY - minY;
        if (height <= 0f) return mapped; // nothing to do

        float shrink = height * 0.1f; // 10%

        // 4) find indices of two largest Y values (bottom-most points)
        float[] ys = new float[4];
        for (int i = 0; i < 4; i++) ys[i] = mapped[i].y;

        int maxIdx1 = -1;
        int maxIdx2 = -1;
        float maxVal1 = Float.MIN_VALUE;
        float maxVal2 = Float.MIN_VALUE;

        for (int i = 0; i < 4; i++) {
            if (ys[i] > maxVal1) {
                maxVal2 = maxVal1;
                maxIdx2 = maxIdx1;
                maxVal1 = ys[i];
                maxIdx1 = i;
            } else if (ys[i] > maxVal2) {
                maxVal2 = ys[i];
                maxIdx2 = i;
            }
        }

        // 5) move only those bottom two points up by 'shrink'
        if (maxIdx1 >= 0) {
            mapped[maxIdx1].y = Math.max(mapped[maxIdx1].y - shrink, minY);
        }
        if (maxIdx2 >= 0) {
            mapped[maxIdx2].y = Math.max(mapped[maxIdx2].y - shrink, minY);
        }

        return mapped;
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
        return expandDocumentCorners(mapped, 0.5f);
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
//        if (currentLightingResult == null) {
//            Toast.makeText(this, "Lighting analysis not ready", Toast.LENGTH_SHORT).show();
//            return;
//        }
//
//        if (currentLightingResult.lightingCondition == LightingAnalyzer.LightingCondition.BAD) {
//            showLightingIssueDialog();
//            return;
//        }
//
//        if (currentBlurResult != null && currentBlurResult.isBlurred) {
//            Toast.makeText(this, "Image is too blurry. Please adjust focus.", Toast.LENGTH_SHORT).show();
//            return;
//        }

        if (cameraManager != null) {
            captureButton.setEnabled(false);
            captureButton.setText("Capturing..");
            cameraManager.captureImage();
        }
    }

//    @SuppressLint("SetTextI18n")
//    private void showLightingIssueDialog() {
//        AlertDialog.Builder builder = new AlertDialog.Builder(this);
//        builder.setTitle("Image Quality Issue")
//                .setMessage("Cannot capture image due to lighting:\n\n" +
//                        generateLightingIssueExplanation(currentLightingResult) +
//                        "\n\nPlease adjust lighting or camera position.")
//                .setPositiveButton("OK", (dialog, which) -> dialog.dismiss())
//                .setNegativeButton("Capture Anyway", (dialog, which) -> {
//                    if (cameraManager != null) {
//                        captureButton.setEnabled(false);
//                        captureButton.setText("Capturing...");
//                        cameraManager.captureImage();
//                    }
//                    dialog.dismiss();
//                })
//                .show();
//    }

    @Override
    public void onLightingAnalyzed(LightingAnalyzer.LightingAnalysisResult result) {
//        currentLightingResult = result;
//        mainHandler.post(() -> {
////            lightingStatusText.setText(result.statusMessage);
////            lightingDetailText.setText(result.detailMessage);
//
//            boolean canCapture = result.isCaptureEnabled
//                    && (currentBlurResult == null || !currentBlurResult.isBlurred);
//            updateCaptureButtonState(canCapture);
//        });

        currentLightingResult = result;
        mainHandler.post(this::updateStarRatingAndStatus);
    }

//    private String generateLightingIssueExplanation(LightingAnalyzer.LightingAnalysisResult result) {
//        StringBuilder explanation = new StringBuilder();
//
//        if (result.hasReflection) {
//            explanation.append("• Reflection detected on document surface\n");
//        }
//        if (result.brightPixelRatio >= 0.55) {
//            explanation.append("• Excessive brightness: possible overexposure\n");
//        }
//
//        if (explanation.length() == 0) {
//            explanation.append("Multiple lighting issues detected");
//        }
//        return explanation.toString();
//    }

    @Override
    public void onShadowDetected(ShadowDetector.ShadowDetectionResult result) {
        currentShadowResult = result; // Store for quality rating
        if (result == null || result.shadowMask == null) return;
        double shadowRatio = result.getShadowRatio();
        mainHandler.post(() -> {
            // Convert shadow mask Mat to Bitmap and display
            Bitmap maskBitmap = matToMaskBitmap(result.shadowMask);
            if (maskBitmap != null) {
                shadowMaskView.setImageBitmap(maskBitmap);
            } else {
                shadowMaskView.setImageBitmap(null);
            }
            // Trigger quality update if needed
            updateStarRatingAndStatus();
        });
    }

    @Override
    public void onImageCaptured(Bitmap bitmap) {
        capturedBitmap = bitmap;

        mainHandler.post(() -> {
            imageView.setImageBitmap(bitmap);

            // CHANGE: Extract ROI from captured image for document detection
            Bitmap roiBitmap = extractROIFromFrame(latestFrame != null ? latestFrame : bitmap);
            if (roiBitmap != null) {
                roiBitmap = ImageUtils.rotateBitmap(roiBitmap , 180);
                Point[] corners = documentDetection.detectDocumentCornersPoints(roiBitmap);
                if (corners != null) {
                    // Warp the document using the full resolution captured image
                    // but scale the corners appropriately
                    Point[] scaledCorners = scaleCornersToCapturedImage(corners, roiBitmap, bitmap); // shrink bottom by 10%
                    scaledCorners = shrinkBottomCorners(scaledCorners, 0.1);
                    Bitmap warpedBitmap = documentDetection.warpToDocumentFromPoints(bitmap, scaledCorners);
                    imageView2.setImageBitmap(warpedBitmap);
                } else {
                    Toast.makeText(this, "No document detected", Toast.LENGTH_SHORT).show();
                    imageView2.setImageResource(R.drawable.no_document);
                }

                // Clean up ROI bitmap if different from original
                if (roiBitmap != bitmap && roiBitmap != latestFrame) {
                    roiBitmap.recycle();
                }
//                roiBitmap = documentDetection.outputCheck(roiBitmap);
//                imageView2.setImageBitmap(roiBitmap);
            }
            showResultsView();

            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled);
            } else {
                updateCaptureButtonState(false);
            }
//            updateCaptureButtonState(true);
        });
    }

    // NEW METHOD: Scale corners from ROI to captured image coordinates
    private Point[] scaleCornersToCapturedImage(Point[] roiCorners, Bitmap roiBitmap, Bitmap capturedBitmap) {
        if (latestFrame == null || overlayView == null) {
            return roiCorners; // fallback
        }
        if (overlayRect == null) {
            return roiCorners; // fallback
        }

        // Scale ROI rect from view coordinates to captured image coordinates
        Rect capturedROI = scaleRectToBitmap(overlayRect,
                capturedBitmap.getWidth(), capturedBitmap.getHeight(),
                overlayView.getWidth(), overlayView.getHeight());

        Point[] scaledCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            // Scale corner from ROI bitmap space to captured image ROI space
            double x = roiCorners[i].x * capturedROI.width() / roiBitmap.getWidth();
            double y = roiCorners[i].y * capturedROI.height() / roiBitmap.getHeight();

            // Translate to captured image coordinates
            scaledCorners[i] = new Point(
                    capturedROI.left + x,
                    capturedROI.top + y
            );
        }
        return scaledCorners;
    }

    private Rect scaleRectToBitmap(RectF rectInView,
                                   int bmpW, int bmpH,
                                   int viewW, int viewH) {

        float scaleX = (float) viewW / bmpW;
        float scaleY = (float) viewH / bmpH;
        // FIT_CENTER ⇒ use the smaller scale
        float scale = Math.min(scaleX, scaleY);

        // padding due to letter-boxing
        float dx = (viewW - bmpW * scale) / 2f;
        float dy = (viewH - bmpH * scale) / 2f;

        // convert rect from view space → bitmap space
        int left   = Math.round((rectInView.left   - dx) / scale);
        int top    = Math.round((rectInView.top    - dy) / scale);
        int right  = Math.round((rectInView.right  - dx) / scale);
        int bottom = Math.round((rectInView.bottom - dy) / scale);

        // clamp to bitmap bounds
        left   = Math.max(0, Math.min(left,   bmpW - 1));
        top    = Math.max(0, Math.min(top,    bmpH - 1));
        right  = Math.max(0, Math.min(right,  bmpW));
        bottom = Math.max(0, Math.min(bottom, bmpH));

        return new Rect(left, top, right, bottom);
    }

    private Point[] shrinkBottomCorners(Point[] corners, double shrinkFactor) {
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

        double shrink = height * shrinkFactor; // e.g. 0.1 = shrink by 10%

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
            if (currentLightingResult != null) {
                updateCaptureButtonState(currentLightingResult.isCaptureEnabled);
            } else {
                updateCaptureButtonState(false);
            }
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
//        currentBlurResult = result;
//        mainHandler.post(() -> {
//            @SuppressLint("DefaultLocale")
////            String blurMessage = String.format("Blur: %b, Occluded: %b, AvgVariance: %.1f, BlurPct: %.3f, OcclusionPct: %.3f)",
////                    result.isBlurred, result.isOccluded,  result.avgVariance, result.blurPercentage, result.occlusionPercentage);
//
////            blurStatusText.setText(blurMessage);
//
//            boolean canCapture = (currentLightingResult != null && currentLightingResult.isCaptureEnabled)
//                    && !result.isBlurred;
//            updateCaptureButtonState(canCapture);
//        });
        currentBlurResult = result;
        mainHandler.post(this::updateStarRatingAndStatus);
    }

    private void updateStarRatingAndStatus() {
        if (currentLightingResult == null || currentBlurResult == null) {
            lightingBlurStatusText.setText("Analyzing...");
            lightingBlurDetailText.setText("");
            setStars(0);
            updateCaptureButtonState(false);
            return;
        }

        // Determine overall quality score (example logic)
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

        // Consider shadow coverage for quality rating
        if (currentShadowResult != null && currentShadowResult.getShadowRatio() > 0.15) { // More than 15% shadow coverage
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

        // Enable capture only if stars >= 2
        updateCaptureButtonState(stars >= 2);
    }

    private void setStars(int count) {
        star1.setImageResource(count >= 1 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
        star2.setImageResource(count >= 2 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
        star3.setImageResource(count >= 3 ? R.drawable.ic_star_filled : R.drawable.ic_star_outline);
    }

    private Bitmap matToMaskBitmap(Mat maskMat) {
        if (maskMat == null || maskMat.empty()) return null;

        try {
            // Create a colored shadow overlay
            Mat rgba = new Mat();
            Imgproc.cvtColor(maskMat, rgba, Imgproc.COLOR_GRAY2RGBA);

            Bitmap bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, bmp);

            // Create a rotated bitmap to match camera orientation
            android.graphics.Matrix matrix = new android.graphics.Matrix();
            matrix.postRotate(180); // Match the 180-degree rotation applied to frames
            Bitmap rotatedBmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);

            // Make shadow areas semi-transparent red for better visibility
            int width = rotatedBmp.getWidth();
            int height = rotatedBmp.getHeight();
            int[] pixels = new int[width * height];
            rotatedBmp.getPixels(pixels, 0, width, 0, 0, width, height);

            for (int i = 0; i < pixels.length; i++) {
                int color = pixels[i];
                int alpha = (color & 0xFF); // grayscale value (0 or 255)
                if (alpha == 0) {
                    pixels[i] = 0x00000000; // fully transparent
                } else {
                    pixels[i] = 0x88FF4444; // semi-transparent red for shadow areas
                }
            }

            rotatedBmp.setPixels(pixels, 0, width, 0, 0, width, height);

            bmp.recycle(); // Clean up original bitmap
            rgba.release();
            return rotatedBmp;
        } catch (Exception e) {
            Log.e(TAG, "Error converting shadow mask Mat to Bitmap", e);
            return null;
        }
    }


}