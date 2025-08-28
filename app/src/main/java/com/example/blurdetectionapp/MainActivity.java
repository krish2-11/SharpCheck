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
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "BlurDetectObjDetect";
    private static final int REQUEST_TAKE_PHOTO = 1001;
    private static final int REQUEST_PICK_IMAGE = 1002;
    private static final int CAMERA_PERMISSION_CODE = 2001;
    private static final int STORAGE_PERMISSION_CODE = 2002;

    // UI Components
    private ImageView originalImageView;
    private ImageView processedImageView;
    private TextView statusText;
    private TextView detectionResults;
    private Button captureButton;
    private Button galleryButton;
    private Button retakeButton;
    private ProgressBar progressBar;

    // Processing classes
    private ImageProcessor imageProcessor;
    private BlurDetector blurDetector;
    private ObjectDetector objectDetector;

    private String currentPhotoPath;
    private Uri selectedImageUri;
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
        blurDetector = new BlurDetector(); // Keep for backward compatibility
        objectDetector = new ObjectDetector();
    }



    private void initializeViews() {
        originalImageView = findViewById(R.id.originalImageView);
        processedImageView = findViewById(R.id.processedImageView);
        statusText = findViewById(R.id.statusText);
        detectionResults = findViewById(R.id.detectionResults);
        captureButton = findViewById(R.id.captureButton);
        galleryButton = findViewById(R.id.galleryButton);
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
        if (galleryButton == null) Log.e(TAG, "galleryButton not found");
        if (retakeButton == null) Log.e(TAG, "retakeButton not found");
        if (progressBar == null) Log.e(TAG, "progressBar not found");
    }

    private void setupClickListeners() {
        if (captureButton != null) {
            captureButton.setOnClickListener(v -> checkCameraPermissionAndCapture());
        }
        if (galleryButton != null) {
            galleryButton.setOnClickListener(v -> checkStoragePermissionAndPickImage());
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

    private void checkStoragePermissionAndPickImage() {
        // For Android 13+ (API 33+), we need READ_MEDIA_IMAGES permission
        // For older versions, we need READ_EXTERNAL_STORAGE
        String permission = android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU
                ? android.Manifest.permission.READ_MEDIA_IMAGES
                : android.Manifest.permission.READ_EXTERNAL_STORAGE;

        if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{permission}, STORAGE_PERMISSION_CODE);
        } else {
            dispatchPickImageIntent();
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
            statusText.setText("Ready to capture or select image");
        }
        if (detectionResults != null) {
            detectionResults.setText("");
        }
        if (captureButton != null) {
            captureButton.setVisibility(View.VISIBLE);
        }
        if (galleryButton != null) {
            galleryButton.setVisibility(View.VISIBLE);
        }
        if (retakeButton != null) {
            retakeButton.setVisibility(View.GONE);
        }
        if (progressBar != null) {
            progressBar.setVisibility(View.GONE);
        }

        // Clear stored paths/URIs
        currentPhotoPath = null;
        selectedImageUri = null;
        originalBitmap = null;
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

    private void dispatchPickImageIntent() {
        Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
        pickImageIntent.setType("image/*");

        // Alternative approach using ACTION_GET_CONTENT for broader compatibility
        Intent getContentIntent = new Intent(Intent.ACTION_GET_CONTENT);
        getContentIntent.setType("image/*");
        getContentIntent.addCategory(Intent.CATEGORY_OPENABLE);

        Intent chooserIntent = Intent.createChooser(getContentIntent, "Select Image");
        chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[]{pickImageIntent});

        try {
            startActivityForResult(chooserIntent, REQUEST_PICK_IMAGE);
        } catch (Exception e) {
            Log.e(TAG, "Error opening gallery", e);
            showToast("Error accessing gallery: " + e.getMessage());
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_TAKE_PHOTO) {
                Log.d(TAG, "Image captured successfully, starting processing");
                selectedImageUri = null; // Clear gallery selection
                processImage();
            } else if (requestCode == REQUEST_PICK_IMAGE && data != null) {
                selectedImageUri = data.getData();
                if (selectedImageUri != null) {
                    Log.d(TAG, "Image selected from gallery: " + selectedImageUri.toString());
                    currentPhotoPath = null; // Clear camera capture path
                    processImage();
                } else {
                    Log.w(TAG, "No image URI received from gallery");
                    showToast("Failed to get image from gallery");
                }
            }
        } else {
            Log.w(TAG, "Image capture/selection cancelled or failed");
            showToast(requestCode == REQUEST_TAKE_PHOTO ? "Image capture cancelled" : "Image selection cancelled");
        }
    }

    private void processImage() {
        showProgressBar(true);
        updateStatus("Loading and processing image...");

        new Thread(() -> {
            try {
                Log.d(TAG, "Starting image processing with blur detection and deblurring");

                // Load and prepare image from either camera or gallery
                if (selectedImageUri != null) {
                    // Image from gallery
                    originalBitmap = imageProcessor.loadAndPrepareImageFromUri(this, selectedImageUri);
                } else if (currentPhotoPath != null) {
                    // Image from camera
                    originalBitmap = imageProcessor.loadAndPrepareImage(currentPhotoPath);
                } else {
                    throw new RuntimeException("No valid image source found");
                }

                if (originalBitmap == null) {
                    throw new RuntimeException("Failed to load image bitmap");
                }

                Log.d(TAG, "Bitmap loaded: " + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());

                runOnUiThread(() -> {
                    if (originalImageView != null) {
                        originalImageView.setImageBitmap(originalBitmap);
                    }
                    updateStatus("Detecting and processing blurred objects...");
                });

                // NEW: Blur detection and deblurring pipeline
                BlurDetectionController blurController = new BlurDetectionController();
                BlurDetectionControllerResult blurResult = blurController.processImage(originalBitmap);

                Log.d(TAG, "Blur processing completed: " + blurResult.getMessage());

                // Get the processed image (either deblurred or original if no blur detected)
                Bitmap processedBitmap = blurResult.getProcessedImage();

                runOnUiThread(() -> {
                    updateStatus("Detecting objects in processed image...");

                    // Display blur processing results
                    displayBlurResults(blurResult);
                });

                // Now perform object detection on the processed (potentially deblurred) image
                Log.d(TAG, "Starting object detection on processed image");
                ObjectDetectionResult detectionResult = objectDetector.detectObjects(processedBitmap);
                Log.d(TAG, "Object detection completed, found: " + detectionResult.getDetectedObjects().size() + " objects");

                runOnUiThread(() -> {
                    displayFinalResults(detectionResult, blurResult);
                });

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

    // NEW: Method to display blur processing results
    private void displayBlurResults(BlurDetectionControllerResult blurResult) {
        if (detectionResults == null) return;

        StringBuilder blurInfo = new StringBuilder();

        if (!blurResult.hadBlurredObjects()) {
            blurInfo.append("✓ Image Quality: Good\n");
            blurInfo.append("• No blurred objects detected\n");
            blurInfo.append("• Ready for object detection\n\n");
        } else {
            blurInfo.append("🔧 Blur Processing Results:\n");
            blurInfo.append(String.format("• Total objects found: %d\n", blurResult.getTotalObjectsDetected()));
            blurInfo.append(String.format("• Blurred objects: %d\n", blurResult.getBlurredObjectsCount()));

            if (blurResult.wasDeblurred()) {
                blurInfo.append("✓ Successfully deblurred objects\n");

                // Show quality improvements if available
                if (blurResult.getDeblurringResult() != null) {
                    List<DeblurredObject> deblurredObjects = blurResult.getDeblurringResult().getDeblurredObjects();
                    if (!deblurredObjects.isEmpty()) {
                        double avgImprovement = 0.0;
                        for (DeblurredObject obj : deblurredObjects) {
                            avgImprovement += obj.getQualityImprovement();
                        }
                        avgImprovement /= deblurredObjects.size();
                        blurInfo.append(String.format("• Average quality improvement: %.1f%%\n", avgImprovement));
                    }
                }
            } else {
                blurInfo.append("⚠ Deblurring partially successful\n");
            }
            blurInfo.append("\n");
        }

        // Store blur info to be combined with object detection results later
        String currentText = detectionResults.getText().toString();
        detectionResults.setText(blurInfo.toString() + currentText);
    }

    // NEW: Enhanced method to display final results combining blur and object detection
    private void displayFinalResults(ObjectDetectionResult objectResult, BlurDetectionControllerResult blurResult) {
        try {
            // Display processed image (potentially deblurred)
            if (objectResult.getProcessedImage() != null && processedImageView != null) {
                processedImageView.setImageBitmap(objectResult.getProcessedImage());
                processedImageView.setVisibility(View.VISIBLE);
            }

            // Combine blur processing and object detection results
            StringBuilder fullResults = new StringBuilder();

            // Blur processing summary
            if (blurResult.hadBlurredObjects()) {
                fullResults.append("🔧 BLUR PROCESSING:\n");
                fullResults.append(String.format("• Objects found: %d (Blurred: %d)\n",
                        blurResult.getTotalObjectsDetected(), blurResult.getBlurredObjectsCount()));

                if (blurResult.wasDeblurred()) {
                    fullResults.append("• Status: ✓ Successfully deblurred\n");

                    // Show detailed blur metrics if available
                    if (blurResult.getDeblurringResult() != null) {
                        List<DeblurredObject> deblurredObjs = blurResult.getDeblurringResult().getDeblurredObjects();
                        for (int i = 0; i < Math.min(3, deblurredObjs.size()); i++) {
                            DeblurredObject obj = deblurredObjs.get(i);
                            BlurredObject original = obj.getOriginalObject();
                            fullResults.append(String.format("  - Object %d: %.1f%% improvement (Method: %s)\n",
                                    i + 1, obj.getQualityImprovement(), original.getDetectionMethod()));
                        }
                        if (deblurredObjs.size() > 3) {
                            fullResults.append(String.format("  - ... and %d more objects\n", deblurredObjs.size() - 3));
                        }
                    }
                } else {
                    fullResults.append("• Status: ⚠ Partial success\n");
                }
                fullResults.append("\n");
            } else {
                fullResults.append("✓ IMAGE QUALITY: Good (No blur detected)\n\n");
            }

            // Object detection results
            fullResults.append("🔍 OBJECT DETECTION:\n");

            if (objectResult.getDetectedObjects().isEmpty()) {
                updateStatus("Processing complete - No objects detected");
                fullResults.append("• No objects detected in processed image\n\n");
                fullResults.append("Tips for better detection:\n");
                fullResults.append("  - Ensure good lighting\n");
                fullResults.append("  - Use plain background\n");
                fullResults.append("  - Keep objects separated\n");
                fullResults.append("  - Ensure objects are in focus\n");
            } else {
                updateStatus(String.format(Locale.getDefault(),
                        "Processing complete - Found %d object(s)", objectResult.getDetectedObjects().size()));

                fullResults.append(String.format("• Successfully detected %d objects:\n\n",
                        objectResult.getDetectedObjects().size()));

                // Show detailed object information
                List<DetectedObject> objects = objectResult.getDetectedObjects();
                for (int i = 0; i < objects.size(); i++) {
                    DetectedObject obj = objects.get(i);
                    fullResults.append(String.format("Object %d:\n", i + 1));
                    fullResults.append(String.format("  • Shape: %s\n", obj.getShapeType()));
                    fullResults.append(String.format("  • Area: %.0f pixels\n", obj.getArea()));
                    fullResults.append(String.format("  • Roundness: %.2f\n", obj.getRoundness()));
                    fullResults.append(String.format("  • Aspect Ratio: %.2f\n", obj.getAspectRatio()));
                    fullResults.append("\n");
                }
            }

            // Processing summary
            fullResults.append("📊 PROCESSING SUMMARY:\n");
            fullResults.append(String.format("• Pipeline: %s\n",
                    blurResult.isSuccess() ? "✓ Successful" : "⚠ Partial"));
            if (blurResult.hadBlurredObjects()) {
                fullResults.append(String.format("• Blur correction: %s\n",
                        blurResult.wasDeblurred() ? "✓ Applied" : "⚠ Limited"));
            }
            fullResults.append(String.format("• Final object count: %d\n", objectResult.getDetectedObjects().size()));

            if (detectionResults != null) {
                detectionResults.setText(fullResults.toString());
            }

            showRetakeButton();
            showProgressBar(false);

        } catch (Exception e) {
            Log.e(TAG, "Error displaying final results", e);
            updateStatus("Error displaying results");
            showProgressBar(false);
        }
    }

    private void handleBlurredImage(BlurResult blurResult) {
        updateStatus("Image is blurred - quality too low for object detection");
        if (detectionResults != null) {
            String blurInfo = String.format(Locale.getDefault(),
                    "Blur Analysis:\n• Laplacian Variance: %.2f\n• Tenengrad Score: %.2f\n• Edge Density: %.2f\n\nImage is too blurry for accurate object detection.\nPlease try with a clearer image.",
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
        if (galleryButton != null) {
            galleryButton.setVisibility(View.GONE);
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

        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (requestCode == CAMERA_PERMISSION_CODE) {
                dispatchTakePictureIntent();
            } else if (requestCode == STORAGE_PERMISSION_CODE) {
                dispatchPickImageIntent();
            }
        } else {
            String message = (requestCode == CAMERA_PERMISSION_CODE)
                    ? "Camera permission is required to take photos"
                    : "Storage permission is required to access gallery";
            showToast(message);
        }
    }
}