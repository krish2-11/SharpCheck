package com.example.blurdetectionapp.utils;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class OverlayView extends View {

    private PointF[] points = null;
    private Bitmap shadowMask = null;
    private Bitmap overlayBitmap = null;
    public enum OverlayType { SQUARE, PORTRAIT, LANDSCAPE }
    private OverlayType overlayType = OverlayType.SQUARE;

    private Paint linePaint;
    private Paint cornerPaint;
    private Paint shadowPaint;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private RectF overlayRect; // set whenever you change the overlay
    private Paint rectPaint;

    public RectF getOverlayRect() {
        return overlayRect;
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
        // Make sure view will draw
        setWillNotDraw(false);

        linePaint = new Paint();
        linePaint.setColor(0xFF00FF00); // Green
        linePaint.setStrokeWidth(6f);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setAntiAlias(true);

        cornerPaint = new Paint();
        cornerPaint.setColor(0xFFFF0000); // Red
        cornerPaint.setStrokeWidth(12f);
        cornerPaint.setStyle(Paint.Style.FILL); // Changed to FILL for better visibility
        cornerPaint.setAntiAlias(true);

        shadowPaint = new Paint();
        shadowPaint.setAlpha(120);
        shadowPaint.setAntiAlias(true);

        rectPaint = new Paint();
        rectPaint.setColor(Color.YELLOW);      // Border color
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(8f);
    }

    public void setPoints(PointF[] points) {
        this.points = points;
        invalidate();
    }

    // alias to match MainActivity
    public void setDocumentCorners(PointF[] points) {
        setPoints(points);
    }

    private RectF computeDestRect(Bitmap bmp) {
        float viewW = getWidth();
        float viewH = getHeight();

        // bitmap aspect
        float bmpRatio = (float) bmp.getWidth() / bmp.getHeight();
        float viewRatio = viewW / viewH;

        float destW, destH;
        if (bmpRatio > viewRatio) {
            // Fit width
            destW = viewW * 0.8f; // or overlayRect.width() if you use it
            destH = destW / bmpRatio;
        } else {
            // Fit height
            destH = viewH * 0.8f;
            destW = destH * bmpRatio;
        }

        float left = (viewW - destW) / 2f;
        float top  = (viewH - destH) / 2f;
        return new RectF(left, top, left + destW, top + destH);
    }


    public void clearCorners() {
        this.points = null;
        this.shadowMask = null;
        invalidate();
    }

    public void setOverlay(Bitmap bitmap, OverlayType type) {
        this.overlayBitmap = bitmap;
        this.overlayType = type;
        calculateOverlayRect();
        invalidate();
    }

    private void calculateOverlayRect() {
        if (overlayBitmap == null) {
            overlayRect = null;
            return;
        }

        switch (overlayType) {
            case SQUARE:
                // hard-coded values in view coordinates (example only)
                overlayRect = new RectF(200f, 400f, 800f, 1000f);
                break;

            case PORTRAIT: // height > width
                overlayRect = new RectF(300f, 200f, 700f, 1200f);
                break;

            case LANDSCAPE: // width > height
                overlayRect = new RectF(150f, 500f, 950f, 900f);
                break;
        }

    }


    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        // Draw shadow mask first (if exists)
        if (shadowMask != null && !shadowMask.isRecycled()) {
            @SuppressLint("DrawAllocation")
            Bitmap scaled = Bitmap.createScaledBitmap(shadowMask, getWidth(), getHeight(), false);
            canvas.drawBitmap(scaled, 0, 0, shadowPaint);
        }

        if (overlayBitmap != null && overlayRect != null) {
            overlayRect = computeDestRect(overlayBitmap);
            canvas.drawBitmap(overlayBitmap, null, overlayRect, paint);
            canvas.drawRect(overlayRect, rectPaint);
        }

        // 2. limit further drawing to overlayRect
        if (overlayRect != null) {
            canvas.save();
            canvas.clipRect(overlayRect);
        }

        // Draw document corners and lines
        if (points != null && points.length == 4) {
            // Draw lines between corners
            for (int i = 0; i < 4; i++) {
                PointF start = points[i];
                PointF end = points[(i + 1) % 4];
                if (start != null && end != null) {
                    canvas.drawLine(start.x, start.y, end.x, end.y, linePaint);
                }
            }

            // Draw corner points (circles for better visibility)
            for (PointF point : points) {
                if (point != null) {
                    canvas.drawCircle(point.x, point.y, 8f, cornerPaint);
                }
            }
        }

        if (overlayRect != null) {
            canvas.restore();   // remove clip
        }
    }
}