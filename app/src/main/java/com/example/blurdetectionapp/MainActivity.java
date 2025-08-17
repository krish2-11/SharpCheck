package com.example.blurdetectionapp;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "BlurDetectObjDetect";
    private static final int REQUEST_TAKE_PHOTO = 1001;
    private static final int CAMERA_PERMISSION_CODE = 2001;

    // UI Components
    private ImageView originalImageView;
    private ImageView processedImageView;
    private TextView statusText;
    private TextView detectionResults;
    private Button captureButton;
    private Button retakeButton;
    private ProgressBar progressBar;

    // Processing classes
    private ImageProcessor imageProcessor;
    private BlurDetector blurDetector;
    private ObjectDetector objectDetector;

    private String currentPhotoPath;
    private Bitmap originalBitmap;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed");
        } else {
            Log.d(TAG, "OpenCV initialized successfully");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeComponents();
        initializeViews();
        setupClickListeners();
    }

    private void initializeComponents() {
        imageProcessor = new ImageProcessor();
        blurDetector = new BlurDetector();
        objectDetector = new ObjectDetector();
    }

    private void initializeViews() {
        originalImageView = findViewById(R.id.originalImageView);
        processedImageView = findViewById(R.id.processedImageView);
        statusText = findViewById(R.id.statusText);
        detectionResults = findViewById(R.id.detectionResults);
        captureButton = findViewById(R.id.captureButton);
        retakeButton = findViewById(R.id.retakeButton);
        progressBar = findViewById(R.id.progressBar);

        // Verify all views are found
        verifyViews();

        // Initially hide retake button and processed image
        if (retakeButton != null) retakeButton.setVisibility(View.GONE);
        if (processedImageView != null) processedImageView.setVisibility(View.GONE);
        if (progressBar != null) progressBar.setVisibility(View.GONE);
    }

    private void verifyViews() {
        if (originalImageView == null) Log.e(TAG, "originalImageView not found");
        if (processedImageView == null) Log.e(TAG, "processedImageView not found");
        if (statusText == null) Log.e(TAG, "statusText not found");
        if (detectionResults == null) Log.e(TAG, "detectionResults not found");
        if (captureButton == null) Log.e(TAG, "captureButton not found");
        if (retakeButton == null) Log.e(TAG, "retakeButton not found");
        if (progressBar == null) Log.e(TAG, "progressBar not found");
    }

    private void setupClickListeners() {
        if (captureButton != null) {
            captureButton.setOnClickListener(v -> checkCameraPermissionAndCapture());
        }
        if (retakeButton != null) {
            retakeButton.setOnClickListener(v -> resetForNewCapture());
        }
    }

    private void checkCameraPermissionAndCapture() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        } else {
            dispatchTakePictureIntent();
        }
    }

    private void resetForNewCapture() {
        if (originalImageView != null) {
            originalImageView.setImageDrawable(null);
        }
        if (processedImageView != null) {
            processedImageView.setImageDrawable(null);
            processedImageView.setVisibility(View.GONE);
        }
        if (statusText != null) {
            statusText.setText("Ready to capture image");
        }
        if (detectionResults != null) {
            detectionResults.setText("");
        }
        if (captureButton != null) {
            captureButton.setVisibility(View.VISIBLE);
        }
        if (retakeButton != null) {
            retakeButton.setVisibility(View.GONE);
        }
        if (progressBar != null) {
            progressBar.setVisibility(View.GONE);
        }
    }

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "CAPTURE_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (storageDir == null || !storageDir.exists()) {
            storageDir = getCacheDir();
        }
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        currentPhotoPath = image.getAbsolutePath();
        Log.d(TAG, "Created image file: " + currentPhotoPath);
        return image;
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                Log.e(TAG, "Error creating image file", ex);
                showToast("Error creating image file: " + ex.getMessage());
                return;
            }

            try {
                Uri photoURI = FileProvider.getUriForFile(this,
                        getPackageName() + ".provider", photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            } catch (Exception e) {
                Log.e(TAG, "Error creating file URI", e);
                showToast("Error accessing camera: " + e.getMessage());
            }
        } else {
            showToast("No camera app available");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_TAKE_PHOTO) {
            if (resultCode == RESULT_OK) {
                Log.d(TAG, "Image captured successfully, starting processing");
                processImage();
            } else {
                Log.w(TAG, "Image capture cancelled or failed");
                showToast("Image capture cancelled");
            }
        }
    }

    private void processImage() {
        showProgressBar(true);
        updateStatus("Loading and processing image...");

        new Thread(() -> {
            try {
                Log.d(TAG, "Starting image processing");

                // Load and prepare image
                originalBitmap = imageProcessor.loadAndPrepareImage(currentPhotoPath);
                if (originalBitmap == null) {
                    throw new RuntimeException("Failed to load image bitmap");
                }

                Log.d(TAG, "Bitmap loaded: " + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());

                runOnUiThread(() -> {
                    if (originalImageView != null) {
                        originalImageView.setImageBitmap(originalBitmap);
                    }
                    updateStatus("Checking image quality...");
                });

                // Check for blur
                BlurResult blurResult = blurDetector.detectBlur(originalBitmap);
                Log.d(TAG, "Blur detection completed: " + blurResult.isBlurred());

                runOnUiThread(() -> {
                    if (blurResult.isBlurred()) {
                        handleBlurredImage(blurResult);
                        return;
                    }
                    updateStatus("Detecting objects...");
                });

                if (!blurResult.isBlurred()) {
                    // Perform object detection
                    Log.d(TAG, "Starting object detection");
                    ObjectDetectionResult detectionResult = objectDetector.detectObjects(originalBitmap);
                    Log.d(TAG, "Object detection completed, found: " + detectionResult.getDetectedObjects().size() + " objects");

                    runOnUiThread(() -> {
                        displayResults(detectionResult);
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Error processing image", e);
                runOnUiThread(() -> {
                    showToast("Error processing image: " + e.getMessage());
                    updateStatus("Processing failed: " + e.getMessage());
                    showProgressBar(false);
                    showRetakeButton();
                });
            }
        }).start();
    }

    private void handleBlurredImage(BlurResult blurResult) {
        updateStatus("Image is blurred - quality too low for object detection");
        if (detectionResults != null) {
            String blurInfo = String.format(Locale.getDefault(),
                    "Blur Analysis:\n• Laplacian Variance: %.2f\n• Tenengrad Score: %.2f\n• Edge Density: %.2f\n\nImage is too blurry for accurate object detection.\nPlease retake with better focus.",
                    blurResult.getLaplacianVariance(),
                    blurResult.getTenengradScore(),
                    blurResult.getEdgeDensity());
            detectionResults.setText(blurInfo);
        }
        showRetakeButton();
        showProgressBar(false);
    }

    private void displayResults(ObjectDetectionResult result) {
        try {
            if (result.getProcessedImage() != null && processedImageView != null) {
                processedImageView.setImageBitmap(result.getProcessedImage());
                processedImageView.setVisibility(View.VISIBLE);
            }

            if (result.getDetectedObjects().isEmpty()) {
                updateStatus("No objects detected");
                if (detectionResults != null) {
                    detectionResults.setText("No significant objects found in the image.\n\nTips for better detection:\n• Ensure good lighting\n• Use plain background\n• Keep objects clearly separated\n• Make sure objects are in focus");
                }
            } else {
                updateStatus(String.format(Locale.getDefault(), "Detected %d object(s)", result.getDetectedObjects().size()));
                displayObjectDetails(result.getDetectedObjects());
            }

            showRetakeButton();
            showProgressBar(false);
        } catch (Exception e) {
            Log.e(TAG, "Error displaying results", e);
            updateStatus("Error displaying results");
            showProgressBar(false);
        }
    }

    private void displayObjectDetails(java.util.List<DetectedObject> objects) {
        if (detectionResults == null) return;

        StringBuilder details = new StringBuilder();
        details.append(String.format(Locale.getDefault(), "Found %d Objects:\n\n", objects.size()));

        for (DetectedObject obj : objects) {
            details.append(String.format(Locale.getDefault(), "Object %d:\n", obj.getId()));
            details.append(String.format(Locale.getDefault(), "• Shape: %s\n", obj.getShapeType()));
            details.append(String.format(Locale.getDefault(), "• Area: %.0f pixels\n", obj.getArea()));
            details.append(String.format(Locale.getDefault(), "• Perimeter: %.1f pixels\n", obj.getPerimeter()));
            details.append(String.format(Locale.getDefault(), "• Roundness: %.2f\n", obj.getRoundness()));
            details.append(String.format(Locale.getDefault(), "• Aspect Ratio: %.2f\n", obj.getAspectRatio()));
            details.append("\n");
        }

        detectionResults.setText(details.toString());
    }

    private void showRetakeButton() {
        if (captureButton != null) {
            captureButton.setVisibility(View.GONE);
        }
        if (retakeButton != null) {
            retakeButton.setVisibility(View.VISIBLE);
        }
    }

    private void showProgressBar(boolean show) {
        if (progressBar != null) {
            progressBar.setVisibility(show ? View.VISIBLE : View.GONE);
        }
    }

    private void updateStatus(String status) {
        if (statusText != null) {
            statusText.setText(status);
        }
        Log.d(TAG, "Status: " + status);
    }

    private void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                dispatchTakePictureIntent();
            } else {
                showToast("Camera permission is required to take photos");
            }
        }
    }
}