package com.example.blurdetectionapp;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PointF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class OverlayView extends View {

    private PointF[] points = null;
    private Bitmap shadowMask = null;

    private Paint linePaint;
    private Paint cornerPaint;
    private Paint shadowPaint;

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        linePaint = new Paint();
        linePaint.setColor(0xFF00FF00); // Green
        linePaint.setStrokeWidth(6f);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setAntiAlias(true);

        cornerPaint = new Paint();
        cornerPaint.setColor(0xFFFF0000); // Red
        cornerPaint.setStrokeWidth(12f);
        cornerPaint.setStyle(Paint.Style.STROKE);
        cornerPaint.setAntiAlias(true);

        shadowPaint = new Paint();
        shadowPaint.setAlpha(120);
        shadowPaint.setAntiAlias(true);
    }

    public void setPoints(PointF[] points) {
        this.points = points;
        invalidate();
    }

    // alias to match MainActivity
    public void setDocumentCorners(PointF[] points) {
        setPoints(points);
    }

    public void setShadowMask(Bitmap mask) {
        this.shadowMask = mask;
        invalidate();
    }

    public void clearCorners() {
        this.points = null;
        invalidate();
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        if (shadowMask != null) {
            @SuppressLint("DrawAllocation")
            Bitmap scaled = Bitmap.createScaledBitmap(shadowMask, getWidth(), getHeight(), false);
            canvas.drawBitmap(scaled, 0, 0, shadowPaint);
        }

        if (points != null && points.length == 4) {
            for (int i = 0; i < 4; i++) {
                PointF start = points[i];
                PointF end = points[(i + 1) % 4];
                canvas.drawLine(start.x, start.y, end.x, end.y, linePaint);
            }
            for (PointF point : points) {
                canvas.drawPoint(point.x, point.y, cornerPaint);
            }
        }
    }
}
