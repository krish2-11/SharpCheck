package com.example.blurdetectionapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST = 100;
    private static final int CAMERA_PERMISSION_CODE = 200;

    ImageView imageView;
    ImageView imageView2;
    TextView resultText;
    Button captureButton;

    Bitmap capturedBitmap; // store last captured image

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Initialization Failed!");
        } else {
            Log.d("OpenCV", "OpenCV Initialized");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        resultText = findViewById(R.id.resultText);
        captureButton = findViewById(R.id.captureButton);

        captureButton.setOnClickListener(v -> askCameraPermission());
    }

    private void askCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        } else {
            openCamera();
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, CAMERA_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && data != null) {
            capturedBitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(capturedBitmap);

            if (isImageBlurred(capturedBitmap)) {
                resultText.setText("Image is Blurred");
            } else {
                resultText.setText("Image is Sharp");
            }

            // Process document detection
            Bitmap processed = detectDocumentCorners(capturedBitmap);
            if (processed != null) {
                imageView2.setImageBitmap(processed);
            } else {
                Toast.makeText(this, "No document detected", Toast.LENGTH_SHORT).show();
            }
        }
    }

    public boolean isImageBlurred(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        Mat gray = new Mat();
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);

        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);

        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(laplacian, mean, stddev);

        double variance = stddev.get(0, 0)[0] * stddev.get(0, 0)[0];
        Log.d("BLUR_DETECTION", "Laplacian Variance: " + variance);
        return variance < 1500;
    }

    private Bitmap detectDocumentCorners(Bitmap bitmap) {
        Mat src = new Mat();
        Utils.bitmapToMat(bitmap, src);

        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 75, 200);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = -1;
        MatOfPoint2f bestContour = null;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 1000) { // ignore small noise
                MatOfPoint2f approxCurve = new MatOfPoint2f();
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * Imgproc.arcLength(contour2f, true), true);

                if (approxCurve.total() == 4 && area > maxArea) {
                    maxArea = area;
                    bestContour = approxCurve;
                }
            }
        }

        if (bestContour != null) {
            Point[] points = bestContour.toArray();

            // Sort points to TL, TR, BR, BL
            Point tl = points[0], tr = points[1], br = points[2], bl = points[3];
            // Simple sorting: based on sum & diff of x+y
            java.util.Arrays.sort(points, (p1, p2) -> Double.compare(p1.y + p1.x, p2.y + p2.x));
            tl = points[0];
            br = points[3];
            java.util.Arrays.sort(points, (p1, p2) -> Double.compare(p1.x - p1.y, p2.x - p2.y));
            tr = points[0];
            bl = points[3];

            double widthTop = Math.hypot(tr.x - tl.x, tr.y - tl.y);
            double widthBottom = Math.hypot(br.x - bl.x, br.y - bl.y);
            double maxWidth = Math.max(widthTop, widthBottom);

            double heightLeft = Math.hypot(bl.x - tl.x, bl.y - tl.y);
            double heightRight = Math.hypot(br.x - tr.x, br.y - tr.y);
            double maxHeight = Math.max(heightLeft, heightRight);

            Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
            srcPoints.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);

            Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);
            dstPoints.put(0, 0,
                    0.0, 0.0,
                    maxWidth - 1, 0.0,
                    maxWidth - 1, maxHeight - 1,
                    0.0, maxHeight - 1);

            Mat warpMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Mat dst = new Mat((int) maxHeight, (int) maxWidth, CvType.CV_8UC4);
            Imgproc.warpPerspective(src, dst, warpMat, dst.size());

            Bitmap output = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dst, output);
            return output;
        }

        return null;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
