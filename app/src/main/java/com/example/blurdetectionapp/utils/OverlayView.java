package com.example.blurdetectionapp.utils;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class OverlayView extends View {

    private PointF[] points = null;
    private Bitmap shadowMask = null;

    public enum OverlayType { SQUARE, PORTRAIT, LANDSCAPE }
    private OverlayType overlayType = OverlayType.SQUARE;

    private Paint linePaint;
    private Paint cornerPaint;
    private Paint shadowPaint;
    private Paint overlayPaint;
    private Paint clearPaint;

    private RectF overlayRect;

    // NEW: Make overlayRect publicly accessible for ROI extraction
    public RectF getOverlayRect() {
        calculateOverlayRect();
        if (overlayRect != null) {
            return new RectF(overlayRect);
        }
        return null;
    }


    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setWillNotDraw(false);

        linePaint = new Paint();
        linePaint.setColor(0xFF00FF00); // Green
        linePaint.setStrokeWidth(6f);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setAntiAlias(true);

        cornerPaint = new Paint();
        cornerPaint.setColor(0xFFFF0000); // Red
        cornerPaint.setStrokeWidth(12f);
        cornerPaint.setStyle(Paint.Style.FILL);
        cornerPaint.setAntiAlias(true);

        shadowPaint = new Paint();
        shadowPaint.setAlpha(120);
        shadowPaint.setAntiAlias(true);

        // Paint for the semi-transparent overlay outside capture area
        overlayPaint = new Paint();
        overlayPaint.setColor(Color.BLACK);
        overlayPaint.setAlpha(150); // Semi-transparent
        overlayPaint.setAntiAlias(true);

        // Paint to clear the capture area (make it transparent)
        clearPaint = new Paint();
        clearPaint.setAntiAlias(true);
        clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
    }

    public void setPoints(PointF[] points) {
        this.points = points;
        invalidate();
    }

    public void setDocumentCorners(PointF[] points) {
        setPoints(points);
    }

    public void clearCorners() {
        this.points = null;
        this.shadowMask = null;
        invalidate();
    }

    public void setShadowMask(Bitmap shadowMask) {
        this.shadowMask = shadowMask;
        invalidate();
    }

    public void setOverlay(Bitmap bitmap, OverlayType type) {
        this.overlayType = type;
        calculateOverlayRect();
        invalidate();
    }

    private void calculateOverlayRect() {
        int viewWidth = getWidth();
        int viewHeight = getHeight();

        if (viewWidth == 0 || viewHeight == 0) {
            // View not measured yet, will be called again in onDraw
            overlayRect = null;
            return;
        }

        float centerX = viewWidth / 2f;
        float centerY = viewHeight / 2f;

        switch (overlayType) {
            case SQUARE:
                // Square overlay (good for documents, receipts)
                float squareSize = Math.min(viewWidth, viewHeight) * 0.7f;
                overlayRect = new RectF(
                        centerX - squareSize/2,
                        centerY - squareSize/2,
                        centerX + squareSize/2,
                        centerY + squareSize/2
                );
                break;

            case PORTRAIT: // A4 Portrait
                // A4 ratio is 1:1.414 (width:height)
                float a4Width = viewWidth * 0.8f;
                float a4Height = a4Width * 1.414f;

                // If height exceeds view bounds, scale down
                if (a4Height > viewHeight * 0.9f) {
                    a4Height = viewHeight * 0.9f;
                    a4Width = a4Height / 1.414f;
                }

                overlayRect = new RectF(
                        centerX - a4Width/2,
                        centerY - a4Height/2,
                        centerX + a4Width/2,
                        centerY + a4Height/2
                );
                break;

            case LANDSCAPE: // Card (Credit Card, Aadhar, PAN, etc.)
                // Standard card ratio is approximately 1.6:1 (85.6mm x 53.98mm)
                float cardWidth = viewWidth * 0.85f;
                float cardHeight = cardWidth / 1.586f; // Credit card ratio

                // Ensure card doesn't exceed view bounds
                if (cardHeight > viewHeight * 0.6f) {
                    cardHeight = viewHeight * 0.6f;
                    cardWidth = cardHeight * 1.586f;
                }

                overlayRect = new RectF(
                        centerX - cardWidth/2,
                        centerY - cardHeight/2,
                        centerX + cardWidth/2,
                        centerY + cardHeight/2
                );
                break;
        }
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        calculateOverlayRect();
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        // Calculate overlay rect if not already calculated
        if (overlayRect == null) {
            calculateOverlayRect();
        }

        // Draw shadow mask first (if exists) - only within overlay area
        if (shadowMask != null && !shadowMask.isRecycled() && overlayRect != null) {
            canvas.save();
            canvas.clipRect(overlayRect);

            @SuppressLint("DrawAllocation")
            Bitmap scaled = Bitmap.createScaledBitmap(shadowMask, getWidth(), getHeight(), false);
            canvas.drawBitmap(scaled, 0, 0, shadowPaint);

            canvas.restore();
        }

        // Create the overlay effect: darken everything except the capture area
        if (overlayRect != null) {
            // Draw semi-transparent overlay over entire view
            canvas.drawRect(0, 0, getWidth(), getHeight(), overlayPaint);

            // Clear the capture area to make it fully visible
            canvas.drawRoundRect(overlayRect, 20f, 20f, clearPaint);

            // Optional: Draw a subtle border around the capture area
            Paint borderPaint = new Paint();
            borderPaint.setColor(Color.WHITE);
            borderPaint.setStyle(Paint.Style.STROKE);
            borderPaint.setStrokeWidth(3f);
            borderPaint.setAntiAlias(true);
            canvas.drawRoundRect(overlayRect, 20f, 20f, borderPaint);
        }

        // Draw document corners and lines only within the capture area
        if (points != null && points.length == 4 && overlayRect != null) {
            canvas.save();
            canvas.clipRect(overlayRect);

            // Draw lines between corners
            for (int i = 0; i < 4; i++) {
                PointF start = points[i];
                PointF end = points[(i + 1) % 4];
                if (start != null && end != null) {
                    canvas.drawLine(start.x, start.y, end.x, end.y, linePaint);
                }
            }

            // Draw corner points
            for (PointF point : points) {
                if (point != null) {
                    canvas.drawCircle(point.x, point.y, 8f, cornerPaint);
                }
            }

            canvas.restore();
        }
    }
}