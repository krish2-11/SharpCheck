package com.example.blurdetectionapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.util.Log;

import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class ImageProcessor {

    private static final String TAG = "ImageProcessor";
    private static final int MAX_IMAGE_DIMENSION = 1024; // Optimized for performance


    // Add this method to your ImageProcessor class

    public Bitmap loadAndPrepareImageFromUri(Context context, Uri imageUri) {
        try {
            // Load bitmap from URI
            InputStream inputStream = context.getContentResolver().openInputStream(imageUri);
            if (inputStream == null) {
                Log.e("ImageProcessor", "Failed to open input stream for URI: " + imageUri);
                return null;
            }

            // Decode bitmap with options to avoid OOM
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeStream(inputStream, null, options);
            inputStream.close();

            // Calculate inSampleSize
            options.inSampleSize = calculateInSampleSize(options, 1024, 1024);
            options.inJustDecodeBounds = false;

            // Decode the actual bitmap
            inputStream = context.getContentResolver().openInputStream(imageUri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream, null, options);
            inputStream.close();

            if (bitmap == null) {
                Log.e("ImageProcessor", "Failed to decode bitmap from URI");
                return null;
            }

            // Apply the same processing as file-based images
            return processLoadedBitmap(bitmap);

        } catch (Exception e) {
            Log.e("ImageProcessor", "Error loading image from URI: " + imageUri, e);
            return null;
        }
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    // Assuming you have a method like this for common processing
    private Bitmap processLoadedBitmap(Bitmap bitmap) {
        // Apply any common processing (rotation correction, scaling, etc.)
        // This should contain the same logic you use in loadAndPrepareImage()
        return bitmap;
    }

    /**
     * Load and prepare image from file path
     * Handles orientation correction and size optimization
     */
    public Bitmap loadAndPrepareImage(String path) {
        try {
            // Check if file exists
            File imageFile = new File(path);
            if (!imageFile.exists()) {
                Log.e(TAG, "Image file not found: " + path);
                return null;
            }

            Log.d(TAG, "Loading image from: " + path);
            Log.d(TAG, "File size: " + imageFile.length() + " bytes");

            // Load with appropriate sampling
            Bitmap bitmap = decodeSampledBitmapFromFile(path, MAX_IMAGE_DIMENSION);
            if (bitmap == null) {
                Log.e(TAG, "Failed to decode bitmap from file");
                return null;
            }

            // Correct orientation
            Bitmap correctedBitmap = correctImageOrientation(bitmap, path);
            Log.d(TAG, "Image prepared successfully: " + correctedBitmap.getWidth() + "x" + correctedBitmap.getHeight());

            return correctedBitmap;

        } catch (Exception e) {
            Log.e(TAG, "Error loading and preparing image", e);
            return null;
        }
    }

    /**
     * Decode bitmap with size optimization to prevent memory issues
     */
    private Bitmap decodeSampledBitmapFromFile(String path, int maxDim) {
        try {
            // First decode to get dimensions
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, options);

            int width = options.outWidth;
            int height = options.outHeight;

            if (width <= 0 || height <= 0) {
                Log.e(TAG, "Invalid image dimensions: " + width + "x" + height);
                return null;
            }

            // Calculate sample size
            int inSampleSize = calculateInSampleSize(width, height, maxDim);

            Log.d(TAG, "Original size: " + width + "x" + height + ", Sample size: " + inSampleSize);

            // Decode with sample size
            BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
            decodeOptions.inSampleSize = inSampleSize;
            decodeOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
            decodeOptions.inMutable = false;

            Bitmap bitmap = BitmapFactory.decodeFile(path, decodeOptions);

            if (bitmap == null) {
                Log.e(TAG, "Failed to decode bitmap with sample size " + inSampleSize);
                return null;
            }

            Log.d(TAG, "Decoded bitmap: " + bitmap.getWidth() + "x" + bitmap.getHeight());
            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "Error decoding bitmap from file", e);
            return null;
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "Out of memory while decoding bitmap", e);
            return null;
        }
    }

    /**
     * Calculate appropriate sample size to fit within max dimension
     */
    private int calculateInSampleSize(int width, int height, int maxDim) {
        int inSampleSize = 1;
        int maxSize = Math.max(width, height);

        while (maxSize / inSampleSize > maxDim) {
            inSampleSize *= 2;
        }

        return inSampleSize;
    }

    /**
     * Correct image orientation based on EXIF data
     */
    private Bitmap correctImageOrientation(Bitmap bitmap, String filePath) {
        if (bitmap == null) return null;

        try {
            ExifInterface exif = new ExifInterface(filePath);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            Matrix matrix = new Matrix();
            boolean needsTransformation = false;

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    matrix.postRotate(90);
                    needsTransformation = true;
                    Log.d(TAG, "Rotating image 90 degrees");
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    matrix.postRotate(180);
                    needsTransformation = true;
                    Log.d(TAG, "Rotating image 180 degrees");
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    matrix.postRotate(270);
                    needsTransformation = true;
                    Log.d(TAG, "Rotating image 270 degrees");
                    break;
                case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                    matrix.preScale(-1.0f, 1.0f);
                    needsTransformation = true;
                    Log.d(TAG, "Flipping image horizontally");
                    break;
                case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                    matrix.preScale(1.0f, -1.0f);
                    needsTransformation = true;
                    Log.d(TAG, "Flipping image vertically");
                    break;
                default:
                    Log.d(TAG, "No orientation correction needed");
                    return bitmap;
            }

            if (needsTransformation) {
                try {
                    Bitmap correctedBitmap = Bitmap.createBitmap(
                            bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                    // Free memory of original bitmap if a new one was created
                    if (correctedBitmap != bitmap && !bitmap.isRecycled()) {
                        bitmap.recycle();
                    }

                    return correctedBitmap;
                } catch (OutOfMemoryError e) {
                    Log.w(TAG, "Out of memory during orientation correction, using original", e);
                    return bitmap;
                }
            }

            return bitmap;

        } catch (IOException e) {
            Log.w(TAG, "Could not read EXIF orientation, using original image", e);
            return bitmap;
        } catch (Exception e) {
            Log.w(TAG, "Error during orientation correction, using original image", e);
            return bitmap;
        }
    }

    /**
     * Check if a bitmap is valid for processing
     */
    public boolean isValidBitmap(Bitmap bitmap) {
        return bitmap != null &&
                !bitmap.isRecycled() &&
                bitmap.getWidth() > 0 &&
                bitmap.getHeight() > 0;
    }

    /**
     * Get image processing statistics
     */
    public String getImageStats(Bitmap bitmap) {
        if (!isValidBitmap(bitmap)) {
            return "Invalid bitmap";
        }

        return String.format("Dimensions: %dx%d, Config: %s, Size: %.2f MB",
                bitmap.getWidth(),
                bitmap.getHeight(),
                bitmap.getConfig(),
                (bitmap.getByteCount() / (1024.0 * 1024.0)));
    }
}