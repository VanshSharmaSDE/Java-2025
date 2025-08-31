package Algorithms.ImageProcessing;

import java.util.*;
import java.util.concurrent.*;
import java.awt.image.BufferedImage;
import java.awt.*;

/**
 * Comprehensive Image Processing Algorithms
 * Filters, transformations, feature detection, and computer vision
 */
public class ImageProcessingAlgorithms {
    
    /**
     * Digital Image Representation
     */
    public static class DigitalImage {
        private final int[][] pixels;
        private final int width, height;
        
        public DigitalImage(int width, int height) {
            this.width = width;
            this.height = height;
            this.pixels = new int[height][width];
        }
        
        public DigitalImage(int[][] pixels) {
            this.height = pixels.length;
            this.width = pixels[0].length;
            this.pixels = new int[height][width];
            for (int i = 0; i < height; i++) {
                System.arraycopy(pixels[i], 0, this.pixels[i], 0, width);
            }
        }
        
        public int getPixel(int x, int y) {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                return pixels[y][x];
            }
            return 0; // Border handling
        }
        
        public void setPixel(int x, int y, int value) {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                pixels[y][x] = Math.max(0, Math.min(255, value));
            }
        }
        
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public int[][] getPixels() { return pixels; }
        
        public DigitalImage copy() {
            return new DigitalImage(pixels);
        }
        
        public void fillRandom() {
            Random random = new Random();
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixels[y][x] = random.nextInt(256);
                }
            }
        }
        
        public void fillGradient() {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixels[y][x] = (x * 255) / width;
                }
            }
        }
        
        public void addCircle(int centerX, int centerY, int radius, int intensity) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int dx = x - centerX;
                    int dy = y - centerY;
                    if (dx * dx + dy * dy <= radius * radius) {
                        pixels[y][x] = intensity;
                    }
                }
            }
        }
        
        public void addNoise(double intensity) {
            Random random = new Random();
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int noise = (int) (random.nextGaussian() * intensity);
                    pixels[y][x] = Math.max(0, Math.min(255, pixels[y][x] + noise));
                }
            }
        }
        
        public double calculateMean() {
            long sum = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    sum += pixels[y][x];
                }
            }
            return (double) sum / (width * height);
        }
        
        public double calculateStandardDeviation() {
            double mean = calculateMean();
            double sumSquaredDiff = 0;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double diff = pixels[y][x] - mean;
                    sumSquaredDiff += diff * diff;
                }
            }
            
            return Math.sqrt(sumSquaredDiff / (width * height));
        }
    }
    
    /**
     * Convolution and Filtering Operations
     */
    public static class ConvolutionFilters {
        
        public static class Kernel {
            private final double[][] weights;
            private final int size;
            private final double divisor;
            
            public Kernel(double[][] weights, double divisor) {
                this.weights = weights;
                this.size = weights.length;
                this.divisor = divisor;
            }
            
            public Kernel(double[][] weights) {
                this(weights, 1.0);
            }
            
            public double getWeight(int x, int y) {
                return weights[y][x] / divisor;
            }
            
            public int getSize() { return size; }
            
            // Common kernels
            public static Kernel GAUSSIAN_BLUR_3X3 = new Kernel(new double[][]{
                {1, 2, 1},
                {2, 4, 2},
                {1, 2, 1}
            }, 16);
            
            public static Kernel GAUSSIAN_BLUR_5X5 = new Kernel(new double[][]{
                {1, 4, 6, 4, 1},
                {4, 16, 24, 16, 4},
                {6, 24, 36, 24, 6},
                {4, 16, 24, 16, 4},
                {1, 4, 6, 4, 1}
            }, 256);
            
            public static Kernel SOBEL_X = new Kernel(new double[][]{
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
            });
            
            public static Kernel SOBEL_Y = new Kernel(new double[][]{
                {-1, -2, -1},
                {0, 0, 0},
                {1, 2, 1}
            });
            
            public static Kernel LAPLACIAN = new Kernel(new double[][]{
                {0, -1, 0},
                {-1, 4, -1},
                {0, -1, 0}
            });
            
            public static Kernel SHARPEN = new Kernel(new double[][]{
                {0, -1, 0},
                {-1, 5, -1},
                {0, -1, 0}
            });
            
            public static Kernel EDGE_DETECT = new Kernel(new double[][]{
                {-1, -1, -1},
                {-1, 8, -1},
                {-1, -1, -1}
            });
        }
        
        public static DigitalImage convolve(DigitalImage image, Kernel kernel) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            int offset = kernel.getSize() / 2;
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    double sum = 0;
                    
                    for (int ky = 0; ky < kernel.getSize(); ky++) {
                        for (int kx = 0; kx < kernel.getSize(); kx++) {
                            int imageX = x + kx - offset;
                            int imageY = y + ky - offset;
                            
                            int pixelValue = image.getPixel(imageX, imageY);
                            sum += pixelValue * kernel.getWeight(kx, ky);
                        }
                    }
                    
                    result.setPixel(x, y, (int) Math.round(sum));
                }
            }
            
            return result;
        }
        
        public static DigitalImage medianFilter(DigitalImage image, int windowSize) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            int offset = windowSize / 2;
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    List<Integer> neighborhood = new ArrayList<>();
                    
                    for (int dy = -offset; dy <= offset; dy++) {
                        for (int dx = -offset; dx <= offset; dx++) {
                            neighborhood.add(image.getPixel(x + dx, y + dy));
                        }
                    }
                    
                    Collections.sort(neighborhood);
                    int median = neighborhood.get(neighborhood.size() / 2);
                    result.setPixel(x, y, median);
                }
            }
            
            return result;
        }
        
        public static DigitalImage bilateralFilter(DigitalImage image, double sigmaSpatial, double sigmaIntensity) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            int windowSize = (int) (2 * sigmaSpatial) | 1; // Ensure odd size
            int offset = windowSize / 2;
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    double weightSum = 0;
                    double intensitySum = 0;
                    int centerIntensity = image.getPixel(x, y);
                    
                    for (int dy = -offset; dy <= offset; dy++) {
                        for (int dx = -offset; dx <= offset; dx++) {
                            int neighborX = x + dx;
                            int neighborY = y + dy;
                            int neighborIntensity = image.getPixel(neighborX, neighborY);
                            
                            // Spatial weight
                            double spatialDist = Math.sqrt(dx * dx + dy * dy);
                            double spatialWeight = Math.exp(-(spatialDist * spatialDist) / (2 * sigmaSpatial * sigmaSpatial));
                            
                            // Intensity weight
                            double intensityDiff = Math.abs(centerIntensity - neighborIntensity);
                            double intensityWeight = Math.exp(-(intensityDiff * intensityDiff) / (2 * sigmaIntensity * sigmaIntensity));
                            
                            double totalWeight = spatialWeight * intensityWeight;
                            weightSum += totalWeight;
                            intensitySum += neighborIntensity * totalWeight;
                        }
                    }
                    
                    result.setPixel(x, y, (int) (intensitySum / weightSum));
                }
            }
            
            return result;
        }
    }
    
    /**
     * Edge Detection Algorithms
     */
    public static class EdgeDetection {
        
        public static class EdgeResult {
            public final DigitalImage magnitude;
            public final DigitalImage direction;
            
            public EdgeResult(DigitalImage magnitude, DigitalImage direction) {
                this.magnitude = magnitude;
                this.direction = direction;
            }
        }
        
        public static EdgeResult sobelEdgeDetection(DigitalImage image) {
            DigitalImage gradX = ConvolutionFilters.convolve(image, ConvolutionFilters.Kernel.SOBEL_X);
            DigitalImage gradY = ConvolutionFilters.convolve(image, ConvolutionFilters.Kernel.SOBEL_Y);
            
            DigitalImage magnitude = new DigitalImage(image.getWidth(), image.getHeight());
            DigitalImage direction = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    double gx = gradX.getPixel(x, y);
                    double gy = gradY.getPixel(x, y);
                    
                    double mag = Math.sqrt(gx * gx + gy * gy);
                    double dir = Math.atan2(gy, gx) * 180 / Math.PI;
                    
                    magnitude.setPixel(x, y, (int) mag);
                    direction.setPixel(x, y, (int) ((dir + 180) * 255 / 360)); // Normalize to 0-255
                }
            }
            
            return new EdgeResult(magnitude, direction);
        }
        
        public static DigitalImage cannyEdgeDetection(DigitalImage image, double lowThreshold, double highThreshold) {
            // Step 1: Gaussian blur to reduce noise
            DigitalImage blurred = ConvolutionFilters.convolve(image, ConvolutionFilters.Kernel.GAUSSIAN_BLUR_5X5);
            
            // Step 2: Compute gradients
            EdgeResult gradients = sobelEdgeDetection(blurred);
            
            // Step 3: Non-maximum suppression
            DigitalImage suppressed = nonMaximumSuppression(gradients.magnitude, gradients.direction);
            
            // Step 4: Double thresholding and edge tracking
            DigitalImage edges = doubleThresholding(suppressed, lowThreshold, highThreshold);
            edges = edgeTrackingByHysteresis(edges, lowThreshold, highThreshold);
            
            return edges;
        }
        
        private static DigitalImage nonMaximumSuppression(DigitalImage magnitude, DigitalImage direction) {
            DigitalImage result = new DigitalImage(magnitude.getWidth(), magnitude.getHeight());
            
            for (int y = 1; y < magnitude.getHeight() - 1; y++) {
                for (int x = 1; x < magnitude.getWidth() - 1; x++) {
                    double angle = (direction.getPixel(x, y) * 360.0 / 255.0) - 180;
                    double mag = magnitude.getPixel(x, y);
                    
                    // Determine neighboring pixels based on gradient direction
                    double neighbor1, neighbor2;
                    
                    if ((angle >= -22.5 && angle < 22.5) || (angle >= 157.5 || angle < -157.5)) {
                        // Horizontal edge
                        neighbor1 = magnitude.getPixel(x - 1, y);
                        neighbor2 = magnitude.getPixel(x + 1, y);
                    } else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) {
                        // Diagonal edge (/)
                        neighbor1 = magnitude.getPixel(x - 1, y + 1);
                        neighbor2 = magnitude.getPixel(x + 1, y - 1);
                    } else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5)) {
                        // Vertical edge
                        neighbor1 = magnitude.getPixel(x, y - 1);
                        neighbor2 = magnitude.getPixel(x, y + 1);
                    } else {
                        // Diagonal edge (\)
                        neighbor1 = magnitude.getPixel(x - 1, y - 1);
                        neighbor2 = magnitude.getPixel(x + 1, y + 1);
                    }
                    
                    // Suppress if not local maximum
                    if (mag >= neighbor1 && mag >= neighbor2) {
                        result.setPixel(x, y, (int) mag);
                    } else {
                        result.setPixel(x, y, 0);
                    }
                }
            }
            
            return result;
        }
        
        private static DigitalImage doubleThresholding(DigitalImage image, double lowThreshold, double highThreshold) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int pixel = image.getPixel(x, y);
                    
                    if (pixel >= highThreshold) {
                        result.setPixel(x, y, 255); // Strong edge
                    } else if (pixel >= lowThreshold) {
                        result.setPixel(x, y, 128); // Weak edge
                    } else {
                        result.setPixel(x, y, 0); // Non-edge
                    }
                }
            }
            
            return result;
        }
        
        private static DigitalImage edgeTrackingByHysteresis(DigitalImage image, double lowThreshold, double highThreshold) {
            DigitalImage result = image.copy();
            boolean changed = true;
            
            while (changed) {
                changed = false;
                
                for (int y = 1; y < image.getHeight() - 1; y++) {
                    for (int x = 1; x < image.getWidth() - 1; x++) {
                        if (result.getPixel(x, y) == 128) { // Weak edge
                            // Check if connected to strong edge
                            boolean connectedToStrong = false;
                            
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    if (result.getPixel(x + dx, y + dy) == 255) {
                                        connectedToStrong = true;
                                        break;
                                    }
                                }
                                if (connectedToStrong) break;
                            }
                            
                            if (connectedToStrong) {
                                result.setPixel(x, y, 255);
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            // Remove remaining weak edges
            for (int y = 0; y < result.getHeight(); y++) {
                for (int x = 0; x < result.getWidth(); x++) {
                    if (result.getPixel(x, y) == 128) {
                        result.setPixel(x, y, 0);
                    }
                }
            }
            
            return result;
        }
    }
    
    /**
     * Morphological Operations
     */
    public static class MorphologicalOperations {
        
        public static class StructuringElement {
            private final boolean[][] kernel;
            private final int centerX, centerY;
            
            public StructuringElement(boolean[][] kernel, int centerX, int centerY) {
                this.kernel = kernel;
                this.centerX = centerX;
                this.centerY = centerY;
            }
            
            public boolean getElement(int x, int y) {
                return x >= 0 && x < kernel[0].length && y >= 0 && y < kernel.length && kernel[y][x];
            }
            
            public int getWidth() { return kernel[0].length; }
            public int getHeight() { return kernel.length; }
            public int getCenterX() { return centerX; }
            public int getCenterY() { return centerY; }
            
            // Common structuring elements
            public static StructuringElement CROSS_3X3 = new StructuringElement(new boolean[][]{
                {false, true, false},
                {true, true, true},
                {false, true, false}
            }, 1, 1);
            
            public static StructuringElement SQUARE_3X3 = new StructuringElement(new boolean[][]{
                {true, true, true},
                {true, true, true},
                {true, true, true}
            }, 1, 1);
            
            public static StructuringElement DIAMOND_5X5 = new StructuringElement(new boolean[][]{
                {false, false, true, false, false},
                {false, true, true, true, false},
                {true, true, true, true, true},
                {false, true, true, true, false},
                {false, false, true, false, false}
            }, 2, 2);
        }
        
        public static DigitalImage erosion(DigitalImage image, StructuringElement se) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    boolean erode = true;
                    
                    // Check all elements in structuring element
                    for (int sy = 0; sy < se.getHeight(); sy++) {
                        for (int sx = 0; sx < se.getWidth(); sx++) {
                            if (se.getElement(sx, sy)) {
                                int imageX = x + sx - se.getCenterX();
                                int imageY = y + sy - se.getCenterY();
                                
                                if (image.getPixel(imageX, imageY) == 0) {
                                    erode = false;
                                    break;
                                }
                            }
                        }
                        if (!erode) break;
                    }
                    
                    result.setPixel(x, y, erode ? 255 : 0);
                }
            }
            
            return result;
        }
        
        public static DigitalImage dilation(DigitalImage image, StructuringElement se) {
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    boolean dilate = false;
                    
                    // Check all elements in structuring element
                    for (int sy = 0; sy < se.getHeight(); sy++) {
                        for (int sx = 0; sx < se.getWidth(); sx++) {
                            if (se.getElement(sx, sy)) {
                                int imageX = x + sx - se.getCenterX();
                                int imageY = y + sy - se.getCenterY();
                                
                                if (image.getPixel(imageX, imageY) > 0) {
                                    dilate = true;
                                    break;
                                }
                            }
                        }
                        if (dilate) break;
                    }
                    
                    result.setPixel(x, y, dilate ? 255 : 0);
                }
            }
            
            return result;
        }
        
        public static DigitalImage opening(DigitalImage image, StructuringElement se) {
            return dilation(erosion(image, se), se);
        }
        
        public static DigitalImage closing(DigitalImage image, StructuringElement se) {
            return erosion(dilation(image, se), se);
        }
        
        public static DigitalImage morphologicalGradient(DigitalImage image, StructuringElement se) {
            DigitalImage dilated = dilation(image, se);
            DigitalImage eroded = erosion(image, se);
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int diff = dilated.getPixel(x, y) - eroded.getPixel(x, y);
                    result.setPixel(x, y, Math.abs(diff));
                }
            }
            
            return result;
        }
        
        public static DigitalImage topHat(DigitalImage image, StructuringElement se) {
            DigitalImage opened = opening(image, se);
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int diff = image.getPixel(x, y) - opened.getPixel(x, y);
                    result.setPixel(x, y, Math.max(0, diff));
                }
            }
            
            return result;
        }
        
        public static DigitalImage blackHat(DigitalImage image, StructuringElement se) {
            DigitalImage closed = closing(image, se);
            DigitalImage result = new DigitalImage(image.getWidth(), image.getHeight());
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int diff = closed.getPixel(x, y) - image.getPixel(x, y);
                    result.setPixel(x, y, Math.max(0, diff));
                }
            }
            
            return result;
        }
    }
    
    /**
     * Feature Detection and Matching
     */
    public static class FeatureDetection {
        
        public static class HarrisCorner {
            public final int x, y;
            public final double response;
            
            public HarrisCorner(int x, int y, double response) {
                this.x = x;
                this.y = y;
                this.response = response;
            }
            
            public String toString() {
                return String.format("Corner[(%d,%d), response=%.3f]", x, y, response);
            }
        }
        
        public static List<HarrisCorner> harrisCornerDetection(DigitalImage image, double threshold, double k) {
            // Compute image gradients
            EdgeDetection.EdgeResult gradients = EdgeDetection.sobelEdgeDetection(image);
            
            List<HarrisCorner> corners = new ArrayList<>();
            int windowSize = 3;
            int offset = windowSize / 2;
            
            for (int y = offset; y < image.getHeight() - offset; y++) {
                for (int x = offset; x < image.getWidth() - offset; x++) {
                    // Compute Harris matrix elements
                    double Ixx = 0, Iyy = 0, Ixy = 0;
                    
                    for (int dy = -offset; dy <= offset; dy++) {
                        for (int dx = -offset; dx <= offset; dx++) {
                            int gx = gradients.magnitude.getPixel(x + dx, y + dy);
                            int gy = gradients.direction.getPixel(x + dx, y + dy);
                            
                            // Convert direction back to actual gradient
                            double angle = (gy * 360.0 / 255.0 - 180) * Math.PI / 180;
                            double gradX = gx * Math.cos(angle);
                            double gradY = gx * Math.sin(angle);
                            
                            Ixx += gradX * gradX;
                            Iyy += gradY * gradY;
                            Ixy += gradX * gradY;
                        }
                    }
                    
                    // Harris corner response
                    double det = Ixx * Iyy - Ixy * Ixy;
                    double trace = Ixx + Iyy;
                    double response = det - k * trace * trace;
                    
                    if (response > threshold) {
                        corners.add(new HarrisCorner(x, y, response));
                    }
                }
            }
            
            // Sort by response strength
            corners.sort((a, b) -> Double.compare(b.response, a.response));
            
            return corners;
        }
        
        public static class Template {
            private final DigitalImage template;
            private final int width, height;
            
            public Template(DigitalImage template) {
                this.template = template;
                this.width = template.getWidth();
                this.height = template.getHeight();
            }
            
            public DigitalImage getTemplate() { return template; }
            public int getWidth() { return width; }
            public int getHeight() { return height; }
        }
        
        public static class MatchResult {
            public final int x, y;
            public final double score;
            
            public MatchResult(int x, int y, double score) {
                this.x = x;
                this.y = y;
                this.score = score;
            }
            
            public String toString() {
                return String.format("Match[(%d,%d), score=%.3f]", x, y, score);
            }
        }
        
        public static MatchResult templateMatching(DigitalImage image, Template template) {
            double bestScore = Double.NEGATIVE_INFINITY;
            int bestX = 0, bestY = 0;
            
            int searchWidth = image.getWidth() - template.getWidth() + 1;
            int searchHeight = image.getHeight() - template.getHeight() + 1;
            
            for (int y = 0; y < searchHeight; y++) {
                for (int x = 0; x < searchWidth; x++) {
                    double score = normalizedCrossCorrelation(image, template, x, y);
                    
                    if (score > bestScore) {
                        bestScore = score;
                        bestX = x;
                        bestY = y;
                    }
                }
            }
            
            return new MatchResult(bestX, bestY, bestScore);
        }
        
        private static double normalizedCrossCorrelation(DigitalImage image, Template template, int offsetX, int offsetY) {
            double imageMean = 0, templateMean = 0;
            int count = template.getWidth() * template.getHeight();
            
            // Calculate means
            for (int y = 0; y < template.getHeight(); y++) {
                for (int x = 0; x < template.getWidth(); x++) {
                    imageMean += image.getPixel(offsetX + x, offsetY + y);
                    templateMean += template.getTemplate().getPixel(x, y);
                }
            }
            imageMean /= count;
            templateMean /= count;
            
            // Calculate correlation
            double numerator = 0, imageSumSq = 0, templateSumSq = 0;
            
            for (int y = 0; y < template.getHeight(); y++) {
                for (int x = 0; x < template.getWidth(); x++) {
                    double imageDiff = image.getPixel(offsetX + x, offsetY + y) - imageMean;
                    double templateDiff = template.getTemplate().getPixel(x, y) - templateMean;
                    
                    numerator += imageDiff * templateDiff;
                    imageSumSq += imageDiff * imageDiff;
                    templateSumSq += templateDiff * templateDiff;
                }
            }
            
            double denominator = Math.sqrt(imageSumSq * templateSumSq);
            return denominator > 0 ? numerator / denominator : 0;
        }
    }
    
    /**
     * Frequency Domain Processing
     */
    public static class FrequencyDomain {
        
        public static class Complex {
            public final double real, imag;
            
            public Complex(double real, double imag) {
                this.real = real;
                this.imag = imag;
            }
            
            public Complex add(Complex other) {
                return new Complex(real + other.real, imag + other.imag);
            }
            
            public Complex subtract(Complex other) {
                return new Complex(real - other.real, imag - other.imag);
            }
            
            public Complex multiply(Complex other) {
                return new Complex(
                    real * other.real - imag * other.imag,
                    real * other.imag + imag * other.real
                );
            }
            
            public double magnitude() {
                return Math.sqrt(real * real + imag * imag);
            }
            
            public double phase() {
                return Math.atan2(imag, real);
            }
            
            public String toString() {
                return String.format("%.3f + %.3fi", real, imag);
            }
        }
        
        public static Complex[][] dft2D(DigitalImage image) {
            int width = image.getWidth();
            int height = image.getHeight();
            Complex[][] result = new Complex[height][width];
            
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {
                    Complex sum = new Complex(0, 0);
                    
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            double angle = -2 * Math.PI * (u * x / (double) width + v * y / (double) height);
                            Complex exponential = new Complex(Math.cos(angle), Math.sin(angle));
                            Complex pixel = new Complex(image.getPixel(x, y), 0);
                            
                            sum = sum.add(pixel.multiply(exponential));
                        }
                    }
                    
                    result[v][u] = sum;
                }
            }
            
            return result;
        }
        
        public static DigitalImage idft2D(Complex[][] spectrum) {
            int height = spectrum.length;
            int width = spectrum[0].length;
            DigitalImage result = new DigitalImage(width, height);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Complex sum = new Complex(0, 0);
                    
                    for (int v = 0; v < height; v++) {
                        for (int u = 0; u < width; u++) {
                            double angle = 2 * Math.PI * (u * x / (double) width + v * y / (double) height);
                            Complex exponential = new Complex(Math.cos(angle), Math.sin(angle));
                            
                            sum = sum.add(spectrum[v][u].multiply(exponential));
                        }
                    }
                    
                    double intensity = sum.real / (width * height);
                    result.setPixel(x, y, (int) Math.round(intensity));
                }
            }
            
            return result;
        }
        
        public static DigitalImage getMagnitudeSpectrum(Complex[][] spectrum) {
            int height = spectrum.length;
            int width = spectrum[0].length;
            DigitalImage result = new DigitalImage(width, height);
            
            // Find max magnitude for normalization
            double maxMagnitude = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double magnitude = Math.log(1 + spectrum[y][x].magnitude());
                    maxMagnitude = Math.max(maxMagnitude, magnitude);
                }
            }
            
            // Normalize and set pixels
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double magnitude = Math.log(1 + spectrum[y][x].magnitude());
                    int intensity = (int) (255 * magnitude / maxMagnitude);
                    result.setPixel(x, y, intensity);
                }
            }
            
            return result;
        }
        
        public static Complex[][] lowPassFilter(Complex[][] spectrum, double cutoffFreq) {
            int height = spectrum.length;
            int width = spectrum[0].length;
            Complex[][] filtered = new Complex[height][width];
            
            int centerX = width / 2;
            int centerY = height / 2;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double distance = Math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    
                    if (distance <= cutoffFreq) {
                        filtered[y][x] = spectrum[y][x];
                    } else {
                        filtered[y][x] = new Complex(0, 0);
                    }
                }
            }
            
            return filtered;
        }
        
        public static Complex[][] highPassFilter(Complex[][] spectrum, double cutoffFreq) {
            int height = spectrum.length;
            int width = spectrum[0].length;
            Complex[][] filtered = new Complex[height][width];
            
            int centerX = width / 2;
            int centerY = height / 2;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double distance = Math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    
                    if (distance >= cutoffFreq) {
                        filtered[y][x] = spectrum[y][x];
                    } else {
                        filtered[y][x] = new Complex(0, 0);
                    }
                }
            }
            
            return filtered;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Image Processing Algorithms Demo:");
        System.out.println("================================");
        
        // Create test image
        DigitalImage testImage = new DigitalImage(64, 64);
        testImage.fillGradient();
        testImage.addCircle(32, 32, 10, 200);
        testImage.addNoise(10);
        
        System.out.println("1. Original Image Statistics:");
        System.out.printf("Mean intensity: %.2f\n", testImage.calculateMean());
        System.out.printf("Standard deviation: %.2f\n", testImage.calculateStandardDeviation());
        
        // Filtering demonstration
        System.out.println("\n2. Image Filtering:");
        
        DigitalImage blurred = ConvolutionFilters.convolve(testImage, ConvolutionFilters.Kernel.GAUSSIAN_BLUR_5X5);
        System.out.printf("Gaussian blur applied. New mean: %.2f\n", blurred.calculateMean());
        
        DigitalImage sharpened = ConvolutionFilters.convolve(testImage, ConvolutionFilters.Kernel.SHARPEN);
        System.out.printf("Sharpening applied. New std dev: %.2f\n", sharpened.calculateStandardDeviation());
        
        DigitalImage medianFiltered = ConvolutionFilters.medianFilter(testImage, 3);
        System.out.printf("Median filter applied. Noise reduction achieved.\n");
        
        DigitalImage bilateral = ConvolutionFilters.bilateralFilter(testImage, 2.0, 50.0);
        System.out.printf("Bilateral filter applied. Edge-preserving smoothing.\n");
        
        // Edge Detection
        System.out.println("\n3. Edge Detection:");
        
        EdgeDetection.EdgeResult sobelResult = EdgeDetection.sobelEdgeDetection(testImage);
        System.out.printf("Sobel edge detection completed.\n");
        
        DigitalImage cannyEdges = EdgeDetection.cannyEdgeDetection(testImage, 50, 150);
        System.out.printf("Canny edge detection completed.\n");
        
        // Morphological Operations
        System.out.println("\n4. Morphological Operations:");
        
        // Create binary test image
        DigitalImage binaryImage = new DigitalImage(32, 32);
        binaryImage.addCircle(16, 16, 8, 255);
        binaryImage.addCircle(10, 10, 3, 255);
        binaryImage.addCircle(22, 22, 3, 255);
        
        DigitalImage eroded = MorphologicalOperations.erosion(binaryImage, 
                                    MorphologicalOperations.StructuringElement.CROSS_3X3);
        System.out.println("Erosion applied to binary image.");
        
        DigitalImage dilated = MorphologicalOperations.dilation(binaryImage, 
                                    MorphologicalOperations.StructuringElement.CROSS_3X3);
        System.out.println("Dilation applied to binary image.");
        
        DigitalImage opened = MorphologicalOperations.opening(binaryImage, 
                                    MorphologicalOperations.StructuringElement.SQUARE_3X3);
        System.out.println("Opening operation completed.");
        
        DigitalImage closed = MorphologicalOperations.closing(binaryImage, 
                                    MorphologicalOperations.StructuringElement.SQUARE_3X3);
        System.out.println("Closing operation completed.");
        
        // Feature Detection
        System.out.println("\n5. Feature Detection:");
        
        List<FeatureDetection.HarrisCorner> corners = 
            FeatureDetection.harrisCornerDetection(testImage, 1000, 0.04);
        System.out.printf("Harris corner detection found %d corners:\n", corners.size());
        for (int i = 0; i < Math.min(5, corners.size()); i++) {
            System.out.printf("  %s\n", corners.get(i));
        }
        
        // Template Matching
        DigitalImage template = new DigitalImage(8, 8);
        template.addCircle(4, 4, 3, 255);
        
        FeatureDetection.Template templateObj = new FeatureDetection.Template(template);
        FeatureDetection.MatchResult matchResult = 
            FeatureDetection.templateMatching(testImage, templateObj);
        System.out.printf("Template matching result: %s\n", matchResult);
        
        // Frequency Domain Processing
        System.out.println("\n6. Frequency Domain Processing:");
        
        // Create smaller image for DFT demonstration
        DigitalImage smallImage = new DigitalImage(16, 16);
        smallImage.fillGradient();
        smallImage.addCircle(8, 8, 4, 200);
        
        FrequencyDomain.Complex[][] spectrum = FrequencyDomain.dft2D(smallImage);
        System.out.println("2D DFT computed.");
        
        DigitalImage magnitudeSpectrum = FrequencyDomain.getMagnitudeSpectrum(spectrum);
        System.out.printf("Magnitude spectrum created. Mean: %.2f\n", magnitudeSpectrum.calculateMean());
        
        // Apply low-pass filter
        FrequencyDomain.Complex[][] filtered = FrequencyDomain.lowPassFilter(spectrum, 5.0);
        DigitalImage reconstructed = FrequencyDomain.idft2D(filtered);
        System.out.printf("Low-pass filtering applied. Reconstructed image mean: %.2f\n", 
                         reconstructed.calculateMean());
        
        // Apply high-pass filter
        FrequencyDomain.Complex[][] highFiltered = FrequencyDomain.highPassFilter(spectrum, 2.0);
        DigitalImage highPassResult = FrequencyDomain.idft2D(highFiltered);
        System.out.printf("High-pass filtering applied. Result std dev: %.2f\n", 
                         highPassResult.calculateStandardDeviation());
        
        System.out.println("\nImage processing demonstration completed!");
        System.out.println("Algorithms demonstrated:");
        System.out.println("- Convolution filtering (Gaussian, Sobel, Laplacian, etc.)");
        System.out.println("- Median and bilateral filtering");
        System.out.println("- Edge detection (Sobel, Canny)");
        System.out.println("- Morphological operations (erosion, dilation, opening, closing)");
        System.out.println("- Feature detection (Harris corners, template matching)");
        System.out.println("- Frequency domain processing (DFT, filtering)");
    }
}
