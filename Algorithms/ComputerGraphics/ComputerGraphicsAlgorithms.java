package Algorithms.ComputerGraphics;

import java.util.*;

/**
 * Comprehensive Computer Graphics Algorithms
 * Rendering, geometry, computer vision, image processing, and 3D graphics
 */
public class ComputerGraphicsAlgorithms {
    
    /**
     * 2D Graphics Primitives
     */
    public static class Graphics2D {
        
        /**
         * Bresenham's Line Drawing Algorithm
         */
        public static List<Point> bresenhamLine(int x0, int y0, int x1, int y1) {
            List<Point> points = new ArrayList<>();
            
            int dx = Math.abs(x1 - x0);
            int dy = Math.abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;
            
            int x = x0, y = y0;
            
            while (true) {
                points.add(new Point(x, y));
                
                if (x == x1 && y == y1) break;
                
                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y += sy;
                }
            }
            
            return points;
        }
        
        /**
         * Bresenham's Circle Drawing Algorithm
         */
        public static List<Point> bresenhamCircle(int centerX, int centerY, int radius) {
            List<Point> points = new ArrayList<>();
            
            int x = 0;
            int y = radius;
            int d = 3 - 2 * radius;
            
            drawCirclePoints(points, centerX, centerY, x, y);
            
            while (y >= x) {
                x++;
                
                if (d > 0) {
                    y--;
                    d = d + 4 * (x - y) + 10;
                } else {
                    d = d + 4 * x + 6;
                }
                
                drawCirclePoints(points, centerX, centerY, x, y);
            }
            
            return points;
        }
        
        private static void drawCirclePoints(List<Point> points, int xc, int yc, int x, int y) {
            points.add(new Point(xc + x, yc + y));
            points.add(new Point(xc - x, yc + y));
            points.add(new Point(xc + x, yc - y));
            points.add(new Point(xc - x, yc - y));
            points.add(new Point(xc + y, yc + x));
            points.add(new Point(xc - y, yc + x));
            points.add(new Point(xc + y, yc - x));
            points.add(new Point(xc - y, yc - x));
        }
        
        /**
         * Flood Fill Algorithm
         */
        public static void floodFill(int[][] image, int x, int y, int newColor, int originalColor) {
            if (x < 0 || x >= image.length || y < 0 || y >= image[0].length) return;
            if (image[x][y] != originalColor || image[x][y] == newColor) return;
            
            image[x][y] = newColor;
            
            // 4-connected fill
            floodFill(image, x + 1, y, newColor, originalColor);
            floodFill(image, x - 1, y, newColor, originalColor);
            floodFill(image, x, y + 1, newColor, originalColor);
            floodFill(image, x, y - 1, newColor, originalColor);
        }
        
        /**
         * Scanline Fill Algorithm
         */
        public static void scanlineFill(int[][] image, List<Point> polygon, int fillColor) {
            if (polygon.size() < 3) return;
            
            // Find bounds
            int minY = polygon.stream().mapToInt(p -> p.y).min().orElse(0);
            int maxY = polygon.stream().mapToInt(p -> p.y).max().orElse(0);
            
            for (int y = minY; y <= maxY; y++) {
                List<Integer> intersections = new ArrayList<>();
                
                // Find intersections with polygon edges
                for (int i = 0; i < polygon.size(); i++) {
                    Point p1 = polygon.get(i);
                    Point p2 = polygon.get((i + 1) % polygon.size());
                    
                    if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
                        double x = p1.x + (double)(y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x);
                        intersections.add((int)x);
                    }
                }
                
                // Sort intersections and fill between pairs
                Collections.sort(intersections);
                for (int i = 0; i < intersections.size(); i += 2) {
                    if (i + 1 < intersections.size()) {
                        int startX = intersections.get(i);
                        int endX = intersections.get(i + 1);
                        
                        for (int x = startX; x <= endX; x++) {
                            if (x >= 0 && x < image[0].length && y >= 0 && y < image.length) {
                                image[y][x] = fillColor;
                            }
                        }
                    }
                }
            }
        }
        
        /**
         * Cohen-Sutherland Line Clipping
         */
        public static class ClippedLine {
            public Point start, end;
            public boolean visible;
            
            public ClippedLine(Point start, Point end, boolean visible) {
                this.start = start;
                this.end = end;
                this.visible = visible;
            }
        }
        
        public static ClippedLine cohenSutherlandClip(Point p1, Point p2, Rectangle clipWindow) {
            int outcode1 = computeOutcode(p1, clipWindow);
            int outcode2 = computeOutcode(p2, clipWindow);
            
            while (true) {
                if ((outcode1 | outcode2) == 0) {
                    // Both points inside
                    return new ClippedLine(p1, p2, true);
                } else if ((outcode1 & outcode2) != 0) {
                    // Both points outside same region
                    return new ClippedLine(p1, p2, false);
                } else {
                    // Line crosses boundary
                    int outcodeOut = outcode1 != 0 ? outcode1 : outcode2;
                    Point intersection = new Point(0, 0);
                    
                    if ((outcodeOut & 8) != 0) { // Top
                        intersection.x = p1.x + (p2.x - p1.x) * (clipWindow.y + clipWindow.height - p1.y) / (p2.y - p1.y);
                        intersection.y = clipWindow.y + clipWindow.height;
                    } else if ((outcodeOut & 4) != 0) { // Bottom
                        intersection.x = p1.x + (p2.x - p1.x) * (clipWindow.y - p1.y) / (p2.y - p1.y);
                        intersection.y = clipWindow.y;
                    } else if ((outcodeOut & 2) != 0) { // Right
                        intersection.y = p1.y + (p2.y - p1.y) * (clipWindow.x + clipWindow.width - p1.x) / (p2.x - p1.x);
                        intersection.x = clipWindow.x + clipWindow.width;
                    } else if ((outcodeOut & 1) != 0) { // Left
                        intersection.y = p1.y + (p2.y - p1.y) * (clipWindow.x - p1.x) / (p2.x - p1.x);
                        intersection.x = clipWindow.x;
                    }
                    
                    if (outcodeOut == outcode1) {
                        p1 = intersection;
                        outcode1 = computeOutcode(p1, clipWindow);
                    } else {
                        p2 = intersection;
                        outcode2 = computeOutcode(p2, clipWindow);
                    }
                }
            }
        }
        
        private static int computeOutcode(Point p, Rectangle clipWindow) {
            int code = 0;
            
            if (p.x < clipWindow.x) code |= 1; // Left
            if (p.x > clipWindow.x + clipWindow.width) code |= 2; // Right
            if (p.y < clipWindow.y) code |= 4; // Bottom
            if (p.y > clipWindow.y + clipWindow.height) code |= 8; // Top
            
            return code;
        }
    }
    
    /**
     * 3D Graphics and Transformations
     */
    public static class Graphics3D {
        
        /**
         * 3D Point representation
         */
        public static class Point3D {
            public double x, y, z, w;
            
            public Point3D(double x, double y, double z) {
                this.x = x;
                this.y = y;
                this.z = z;
                this.w = 1.0;
            }
            
            public Point3D(double x, double y, double z, double w) {
                this.x = x;
                this.y = y;
                this.z = z;
                this.w = w;
            }
            
            @Override
            public String toString() {
                return String.format("(%.2f, %.2f, %.2f)", x, y, z);
            }
        }
        
        /**
         * 4x4 Transformation Matrix
         */
        public static class Matrix4x4 {
            public double[][] m = new double[4][4];
            
            public Matrix4x4() {
                // Identity matrix
                for (int i = 0; i < 4; i++) {
                    m[i][i] = 1.0;
                }
            }
            
            public Matrix4x4(double[][] matrix) {
                for (int i = 0; i < 4; i++) {
                    System.arraycopy(matrix[i], 0, m[i], 0, 4);
                }
            }
            
            public Point3D multiply(Point3D point) {
                double x = m[0][0] * point.x + m[0][1] * point.y + m[0][2] * point.z + m[0][3] * point.w;
                double y = m[1][0] * point.x + m[1][1] * point.y + m[1][2] * point.z + m[1][3] * point.w;
                double z = m[2][0] * point.x + m[2][1] * point.y + m[2][2] * point.z + m[2][3] * point.w;
                double w = m[3][0] * point.x + m[3][1] * point.y + m[3][2] * point.z + m[3][3] * point.w;
                
                return new Point3D(x, y, z, w);
            }
            
            public Matrix4x4 multiply(Matrix4x4 other) {
                Matrix4x4 result = new Matrix4x4();
                
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        result.m[i][j] = 0;
                        for (int k = 0; k < 4; k++) {
                            result.m[i][j] += m[i][k] * other.m[k][j];
                        }
                    }
                }
                
                return result;
            }
        }
        
        /**
         * Create translation matrix
         */
        public static Matrix4x4 createTranslation(double tx, double ty, double tz) {
            Matrix4x4 matrix = new Matrix4x4();
            matrix.m[0][3] = tx;
            matrix.m[1][3] = ty;
            matrix.m[2][3] = tz;
            return matrix;
        }
        
        /**
         * Create rotation matrix around X axis
         */
        public static Matrix4x4 createRotationX(double angle) {
            Matrix4x4 matrix = new Matrix4x4();
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            
            matrix.m[1][1] = cos;
            matrix.m[1][2] = -sin;
            matrix.m[2][1] = sin;
            matrix.m[2][2] = cos;
            
            return matrix;
        }
        
        /**
         * Create rotation matrix around Y axis
         */
        public static Matrix4x4 createRotationY(double angle) {
            Matrix4x4 matrix = new Matrix4x4();
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            
            matrix.m[0][0] = cos;
            matrix.m[0][2] = sin;
            matrix.m[2][0] = -sin;
            matrix.m[2][2] = cos;
            
            return matrix;
        }
        
        /**
         * Create rotation matrix around Z axis
         */
        public static Matrix4x4 createRotationZ(double angle) {
            Matrix4x4 matrix = new Matrix4x4();
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            
            matrix.m[0][0] = cos;
            matrix.m[0][1] = -sin;
            matrix.m[1][0] = sin;
            matrix.m[1][1] = cos;
            
            return matrix;
        }
        
        /**
         * Create scaling matrix
         */
        public static Matrix4x4 createScale(double sx, double sy, double sz) {
            Matrix4x4 matrix = new Matrix4x4();
            matrix.m[0][0] = sx;
            matrix.m[1][1] = sy;
            matrix.m[2][2] = sz;
            return matrix;
        }
        
        /**
         * Create perspective projection matrix
         */
        public static Matrix4x4 createPerspective(double fov, double aspect, double near, double far) {
            Matrix4x4 matrix = new Matrix4x4();
            double f = 1.0 / Math.tan(fov / 2.0);
            
            matrix.m[0][0] = f / aspect;
            matrix.m[1][1] = f;
            matrix.m[2][2] = (far + near) / (near - far);
            matrix.m[2][3] = (2.0 * far * near) / (near - far);
            matrix.m[3][2] = -1.0;
            matrix.m[3][3] = 0.0;
            
            return matrix;
        }
        
        /**
         * Z-Buffer Algorithm for Hidden Surface Removal
         */
        public static class ZBuffer {
            private double[][] zBuffer;
            private int[][] frameBuffer;
            private int width, height;
            
            public ZBuffer(int width, int height) {
                this.width = width;
                this.height = height;
                this.zBuffer = new double[height][width];
                this.frameBuffer = new int[height][width];
                
                // Initialize z-buffer to far distance
                for (int i = 0; i < height; i++) {
                    Arrays.fill(zBuffer[i], Double.MAX_VALUE);
                }
            }
            
            public void drawTriangle(Point3D p1, Point3D p2, Point3D p3, int color) {
                // Convert to screen coordinates and draw triangle with z-buffering
                drawTriangleWithZBuffer(p1, p2, p3, color);
            }
            
            private void drawTriangleWithZBuffer(Point3D p1, Point3D p2, Point3D p3, int color) {
                // Bounding box
                int minX = (int) Math.max(0, Math.min(Math.min(p1.x, p2.x), p3.x));
                int maxX = (int) Math.min(width - 1, Math.max(Math.max(p1.x, p2.x), p3.x));
                int minY = (int) Math.max(0, Math.min(Math.min(p1.y, p2.y), p3.y));
                int maxY = (int) Math.min(height - 1, Math.max(Math.max(p1.y, p2.y), p3.y));
                
                for (int y = minY; y <= maxY; y++) {
                    for (int x = minX; x <= maxX; x++) {
                        double[] barycentric = calculateBarycentric(new Point(x, y), p1, p2, p3);
                        
                        if (barycentric[0] >= 0 && barycentric[1] >= 0 && barycentric[2] >= 0) {
                            double z = barycentric[0] * p1.z + barycentric[1] * p2.z + barycentric[2] * p3.z;
                            
                            if (z < zBuffer[y][x]) {
                                zBuffer[y][x] = z;
                                frameBuffer[y][x] = color;
                            }
                        }
                    }
                }
            }
            
            private double[] calculateBarycentric(Point p, Point3D p1, Point3D p2, Point3D p3) {
                double denominator = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);
                double a = ((p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y)) / denominator;
                double b = ((p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y)) / denominator;
                double c = 1 - a - b;
                
                return new double[]{a, b, c};
            }
            
            public int[][] getFrameBuffer() {
                return frameBuffer;
            }
        }
    }
    
    /**
     * Ray Tracing
     */
    public static class RayTracing {
        
        public static class Vector3D {
            public double x, y, z;
            
            public Vector3D(double x, double y, double z) {
                this.x = x;
                this.y = y;
                this.z = z;
            }
            
            public Vector3D add(Vector3D other) {
                return new Vector3D(x + other.x, y + other.y, z + other.z);
            }
            
            public Vector3D subtract(Vector3D other) {
                return new Vector3D(x - other.x, y - other.y, z - other.z);
            }
            
            public Vector3D multiply(double scalar) {
                return new Vector3D(x * scalar, y * scalar, z * scalar);
            }
            
            public double dot(Vector3D other) {
                return x * other.x + y * other.y + z * other.z;
            }
            
            public Vector3D cross(Vector3D other) {
                return new Vector3D(
                    y * other.z - z * other.y,
                    z * other.x - x * other.z,
                    x * other.y - y * other.x
                );
            }
            
            public double length() {
                return Math.sqrt(x * x + y * y + z * z);
            }
            
            public Vector3D normalize() {
                double len = length();
                return new Vector3D(x / len, y / len, z / len);
            }
        }
        
        public static class Ray {
            public Vector3D origin, direction;
            
            public Ray(Vector3D origin, Vector3D direction) {
                this.origin = origin;
                this.direction = direction.normalize();
            }
            
            public Vector3D pointAt(double t) {
                return origin.add(direction.multiply(t));
            }
        }
        
        public static class Sphere {
            public Vector3D center;
            public double radius;
            public Color color;
            
            public Sphere(Vector3D center, double radius, Color color) {
                this.center = center;
                this.radius = radius;
                this.color = color;
            }
            
            public double intersect(Ray ray) {
                Vector3D oc = ray.origin.subtract(center);
                double a = ray.direction.dot(ray.direction);
                double b = 2.0 * oc.dot(ray.direction);
                double c = oc.dot(oc) - radius * radius;
                
                double discriminant = b * b - 4 * a * c;
                
                if (discriminant < 0) {
                    return -1; // No intersection
                }
                
                double t1 = (-b - Math.sqrt(discriminant)) / (2 * a);
                double t2 = (-b + Math.sqrt(discriminant)) / (2 * a);
                
                if (t1 > 0) return t1;
                if (t2 > 0) return t2;
                return -1;
            }
            
            public Vector3D getNormal(Vector3D point) {
                return point.subtract(center).normalize();
            }
        }
        
        public static class Color {
            public double r, g, b;
            
            public Color(double r, double g, double b) {
                this.r = Math.max(0, Math.min(1, r));
                this.g = Math.max(0, Math.min(1, g));
                this.b = Math.max(0, Math.min(1, b));
            }
            
            public Color add(Color other) {
                return new Color(r + other.r, g + other.g, b + other.b);
            }
            
            public Color multiply(double scalar) {
                return new Color(r * scalar, g * scalar, b * scalar);
            }
            
            public Color multiply(Color other) {
                return new Color(r * other.r, g * other.g, b * other.b);
            }
        }
        
        public static class Light {
            public Vector3D position;
            public Color color;
            public double intensity;
            
            public Light(Vector3D position, Color color, double intensity) {
                this.position = position;
                this.color = color;
                this.intensity = intensity;
            }
        }
        
        public static class Scene {
            public List<Sphere> spheres = new ArrayList<>();
            public List<Light> lights = new ArrayList<>();
            public Color backgroundColor = new Color(0.1, 0.1, 0.2);
            
            public Color trace(Ray ray, int depth) {
                if (depth <= 0) return backgroundColor;
                
                double closestT = Double.MAX_VALUE;
                Sphere closestSphere = null;
                
                // Find closest intersection
                for (Sphere sphere : spheres) {
                    double t = sphere.intersect(ray);
                    if (t > 0 && t < closestT) {
                        closestT = t;
                        closestSphere = sphere;
                    }
                }
                
                if (closestSphere == null) {
                    return backgroundColor;
                }
                
                Vector3D hitPoint = ray.pointAt(closestT);
                Vector3D normal = closestSphere.getNormal(hitPoint);
                
                return calculateLighting(hitPoint, normal, closestSphere.color, ray.direction);
            }
            
            private Color calculateLighting(Vector3D point, Vector3D normal, Color materialColor, Vector3D viewDir) {
                Color finalColor = new Color(0, 0, 0);
                
                for (Light light : lights) {
                    Vector3D lightDir = light.position.subtract(point).normalize();
                    double distance = light.position.subtract(point).length();
                    
                    // Check for shadows
                    Ray shadowRay = new Ray(point.add(normal.multiply(0.001)), lightDir);
                    boolean inShadow = false;
                    
                    for (Sphere sphere : spheres) {
                        double t = sphere.intersect(shadowRay);
                        if (t > 0 && t < distance) {
                            inShadow = true;
                            break;
                        }
                    }
                    
                    if (!inShadow) {
                        // Diffuse lighting
                        double diffuse = Math.max(0, normal.dot(lightDir));
                        Color diffuseColor = materialColor.multiply(light.color).multiply(diffuse * light.intensity);
                        
                        // Specular lighting (Phong model)
                        Vector3D reflectDir = lightDir.subtract(normal.multiply(2 * lightDir.dot(normal)));
                        double specular = Math.pow(Math.max(0, reflectDir.dot(viewDir.multiply(-1))), 32);
                        Color specularColor = light.color.multiply(specular * light.intensity * 0.5);
                        
                        finalColor = finalColor.add(diffuseColor).add(specularColor);
                    }
                }
                
                // Ambient lighting
                Color ambient = materialColor.multiply(0.1);
                finalColor = finalColor.add(ambient);
                
                return finalColor;
            }
        }
        
        public static Color[][] renderScene(Scene scene, int width, int height, Vector3D cameraPos, Vector3D lookAt) {
            Color[][] image = new Color[height][width];
            
            Vector3D forward = lookAt.subtract(cameraPos).normalize();
            Vector3D right = forward.cross(new Vector3D(0, 1, 0)).normalize();
            Vector3D up = right.cross(forward).normalize();
            
            double aspectRatio = (double) width / height;
            double fov = Math.PI / 4; // 45 degrees
            double viewPlaneHalfWidth = Math.tan(fov / 2);
            double viewPlaneHalfHeight = viewPlaneHalfWidth / aspectRatio;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double u = (2.0 * x / width - 1.0) * viewPlaneHalfWidth;
                    double v = (1.0 - 2.0 * y / height) * viewPlaneHalfHeight;
                    
                    Vector3D rayDir = forward.add(right.multiply(u)).add(up.multiply(v)).normalize();
                    Ray ray = new Ray(cameraPos, rayDir);
                    
                    image[y][x] = scene.trace(ray, 5);
                }
            }
            
            return image;
        }
    }
    
    /**
     * Image Processing Algorithms
     */
    public static class ImageProcessing {
        
        /**
         * Gaussian Blur
         */
        public static double[][] gaussianBlur(double[][] image, double sigma) {
            int size = (int) (6 * sigma) | 1; // Ensure odd size
            double[][] kernel = createGaussianKernel(size, sigma);
            
            return convolve(image, kernel);
        }
        
        private static double[][] createGaussianKernel(int size, double sigma) {
            double[][] kernel = new double[size][size];
            double sum = 0;
            int center = size / 2;
            
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    int x = i - center;
                    int y = j - center;
                    kernel[i][j] = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
                    sum += kernel[i][j];
                }
            }
            
            // Normalize
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    kernel[i][j] /= sum;
                }
            }
            
            return kernel;
        }
        
        /**
         * Edge Detection using Sobel operator
         */
        public static double[][] sobelEdgeDetection(double[][] image) {
            double[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            double[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
            
            double[][] gradX = convolve(image, sobelX);
            double[][] gradY = convolve(image, sobelY);
            
            int height = image.length;
            int width = image[0].length;
            double[][] result = new double[height][width];
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    result[i][j] = Math.sqrt(gradX[i][j] * gradX[i][j] + gradY[i][j] * gradY[i][j]);
                }
            }
            
            return result;
        }
        
        /**
         * Convolution operation
         */
        public static double[][] convolve(double[][] image, double[][] kernel) {
            int height = image.length;
            int width = image[0].length;
            int kHeight = kernel.length;
            int kWidth = kernel[0].length;
            int kCenterY = kHeight / 2;
            int kCenterX = kWidth / 2;
            
            double[][] result = new double[height][width];
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    double sum = 0;
                    
                    for (int ki = 0; ki < kHeight; ki++) {
                        for (int kj = 0; kj < kWidth; kj++) {
                            int ii = i + ki - kCenterY;
                            int jj = j + kj - kCenterX;
                            
                            // Handle borders by clamping
                            ii = Math.max(0, Math.min(height - 1, ii));
                            jj = Math.max(0, Math.min(width - 1, jj));
                            
                            sum += image[ii][jj] * kernel[ki][kj];
                        }
                    }
                    
                    result[i][j] = sum;
                }
            }
            
            return result;
        }
        
        /**
         * Histogram Equalization
         */
        public static double[][] histogramEqualization(double[][] image) {
            int height = image.length;
            int width = image[0].length;
            int totalPixels = height * width;
            
            // Calculate histogram
            int[] histogram = new int[256];
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int intensity = (int) (image[i][j] * 255);
                    histogram[intensity]++;
                }
            }
            
            // Calculate cumulative distribution function
            int[] cdf = new int[256];
            cdf[0] = histogram[0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i - 1] + histogram[i];
            }
            
            // Normalize and create lookup table
            double[] lookupTable = new double[256];
            int cdfMin = cdf[0];
            for (int i = 0; i < 256; i++) {
                lookupTable[i] = (double) (cdf[i] - cdfMin) / (totalPixels - cdfMin);
            }
            
            // Apply equalization
            double[][] result = new double[height][width];
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int intensity = (int) (image[i][j] * 255);
                    result[i][j] = lookupTable[intensity];
                }
            }
            
            return result;
        }
        
        /**
         * Morphological operations
         */
        public static double[][] morphologicalErosion(double[][] image, boolean[][] structuringElement) {
            int height = image.length;
            int width = image[0].length;
            int seHeight = structuringElement.length;
            int seWidth = structuringElement[0].length;
            int seCenterY = seHeight / 2;
            int seCenterX = seWidth / 2;
            
            double[][] result = new double[height][width];
            
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    double minValue = Double.MAX_VALUE;
                    
                    for (int ki = 0; ki < seHeight; ki++) {
                        for (int kj = 0; kj < seWidth; kj++) {
                            if (structuringElement[ki][kj]) {
                                int ii = i + ki - seCenterY;
                                int jj = j + kj - seCenterX;
                                
                                if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
                                    minValue = Math.min(minValue, image[ii][jj]);
                                }
                            }
                        }
                    }
                    
                    result[i][j] = minValue == Double.MAX_VALUE ? 0 : minValue;
                }
            }
            
            return result;
        }
    }
    
    // Helper classes
    public static class Point {
        public int x, y;
        
        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
        
        @Override
        public String toString() {
            return "(" + x + ", " + y + ")";
        }
    }
    
    public static class Rectangle {
        public int x, y, width, height;
        
        public Rectangle(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Computer Graphics Algorithms Demo:");
        System.out.println("==================================");
        
        // 2D Graphics Demo
        System.out.println("1. 2D Graphics Primitives:");
        List<Point> linePoints = Graphics2D.bresenhamLine(0, 0, 10, 5);
        System.out.println("Bresenham line points: " + linePoints.subList(0, Math.min(5, linePoints.size())));
        
        List<Point> circlePoints = Graphics2D.bresenhamCircle(5, 5, 3);
        System.out.println("Circle points count: " + circlePoints.size());
        
        // Line clipping demo
        Graphics2D.ClippedLine clipped = Graphics2D.cohenSutherlandClip(
            new Point(-5, -5), new Point(15, 15), new Rectangle(0, 0, 10, 10));
        System.out.println("Line clipping result: " + (clipped.visible ? "Visible" : "Not visible"));
        
        // 3D Transformations Demo
        System.out.println("\n2. 3D Transformations:");
        Graphics3D.Point3D point = new Graphics3D.Point3D(1, 2, 3);
        Graphics3D.Matrix4x4 translation = Graphics3D.createTranslation(5, 0, 0);
        Graphics3D.Matrix4x4 rotation = Graphics3D.createRotationZ(Math.PI / 4);
        
        Graphics3D.Point3D transformed = translation.multiply(rotation.multiply(point));
        System.out.println("Original point: " + point);
        System.out.println("Transformed point: " + transformed);
        
        // Ray Tracing Demo
        System.out.println("\n3. Ray Tracing:");
        RayTracing.Scene scene = new RayTracing.Scene();
        
        // Add sphere to scene
        scene.spheres.add(new RayTracing.Sphere(
            new RayTracing.Vector3D(0, 0, -5),
            1.0,
            new RayTracing.Color(1, 0, 0)
        ));
        
        // Add light
        scene.lights.add(new RayTracing.Light(
            new RayTracing.Vector3D(-2, 2, -3),
            new RayTracing.Color(1, 1, 1),
            1.0
        ));
        
        // Render small image
        RayTracing.Color[][] image = RayTracing.renderScene(
            scene, 4, 4,
            new RayTracing.Vector3D(0, 0, 0),
            new RayTracing.Vector3D(0, 0, -1)
        );
        
        System.out.println("Rendered 4x4 image:");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                RayTracing.Color pixel = image[i][j];
                System.out.printf("(%.1f,%.1f,%.1f) ", pixel.r, pixel.g, pixel.b);
            }
            System.out.println();
        }
        
        // Image Processing Demo
        System.out.println("\n4. Image Processing:");
        
        // Create sample image
        double[][] sampleImage = {
            {0.1, 0.2, 0.3, 0.4},
            {0.2, 0.8, 0.9, 0.3},
            {0.3, 0.9, 1.0, 0.2},
            {0.4, 0.3, 0.2, 0.1}
        };
        
        // Apply Gaussian blur
        double[][] blurred = ImageProcessing.gaussianBlur(sampleImage, 1.0);
        System.out.println("Original image center: " + sampleImage[1][1]);
        System.out.println("Blurred image center: " + String.format("%.3f", blurred[1][1]));
        
        // Edge detection
        double[][] edges = ImageProcessing.sobelEdgeDetection(sampleImage);
        System.out.println("Edge strength at center: " + String.format("%.3f", edges[1][1]));
        
        // Z-Buffer Demo
        System.out.println("\n5. Hidden Surface Removal (Z-Buffer):");
        Graphics3D.ZBuffer zBuffer = new Graphics3D.ZBuffer(10, 10);
        
        // Draw triangle
        Graphics3D.Point3D p1 = new Graphics3D.Point3D(2, 2, 5);
        Graphics3D.Point3D p2 = new Graphics3D.Point3D(7, 2, 5);
        Graphics3D.Point3D p3 = new Graphics3D.Point3D(5, 7, 5);
        
        zBuffer.drawTriangle(p1, p2, p3, 255);
        System.out.println("Z-buffer triangle rendering completed");
        
        // Performance metrics
        System.out.println("\n6. Performance Metrics:");
        long startTime = System.nanoTime();
        
        // Benchmark line drawing
        for (int i = 0; i < 1000; i++) {
            Graphics2D.bresenhamLine(0, 0, 100, 100);
        }
        
        long endTime = System.nanoTime();
        System.out.println("1000 line drawings took: " + (endTime - startTime) / 1_000_000 + " ms");
        
        System.out.println("\nGraphics algorithms demonstration completed!");
    }
}
