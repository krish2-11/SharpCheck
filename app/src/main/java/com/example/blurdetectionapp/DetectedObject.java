package com.example.blurdetectionapp;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;

public class DetectedObject {

    private final int id;
    private final double area;
    private final double perimeter;
    private final double roundness;
    private final double aspectRatio;
    private final Point center;
    private final Rect boundingRect;
    private final String shapeType;
    private final MatOfPoint contour;

    public DetectedObject(int id, double area, double perimeter, double roundness,
                          double aspectRatio, Point center, Rect boundingRect,
                          String shapeType, MatOfPoint c) {
        this.id = id;
        this.area = area;
        this.perimeter = perimeter;
        this.roundness = roundness;
        this.aspectRatio = aspectRatio;
        this.center = center;
        this.boundingRect = boundingRect;
        this.shapeType = shapeType;
//        this.contour = (MatOfPoint) c.clone(); // Clone to avoid external modifications
        this.contour = new MatOfPoint();
        Point[] points = c.toArray();
        this.contour.fromArray(points);
    }

    // Getters
    public int getId() {
        return id;
    }

    public double getArea() {
        return area;
    }

    public double getPerimeter() {
        return perimeter;
    }

    public double getRoundness() {
        return roundness;
    }

    public double getAspectRatio() {
        return aspectRatio;
    }

    public Point getCenter() {
        return center;
    }

    public Rect getBoundingRect() {
        return boundingRect;
    }

    public String getShapeType() {
        return shapeType;
    }

    public MatOfPoint getContour() {
        return contour;
    }

    @Override
    public String toString() {
        return String.format("DetectedObject{id=%d, shape='%s', area=%.1f, perimeter=%.1f, roundness=%.3f, aspectRatio=%.2f}",
                id, shapeType, area, perimeter, roundness, aspectRatio);
    }
}