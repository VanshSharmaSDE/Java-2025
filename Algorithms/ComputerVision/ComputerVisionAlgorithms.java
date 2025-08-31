package Algorithms.ComputerVision;

import java.util.*;
import java.awt.image.BufferedImage;

/**
 * Advanced Computer Vision and Machine Learning Algorithms
 * Object detection, feature matching, stereo vision, deep learning
 */
public class ComputerVisionAlgorithms {
    
    /**
     * Feature Detection and Description Algorithms
     */
    public static class FeatureDetection {
        
        public static class Keypoint {
            public final double x, y;
            public final double scale;
            public final double orientation;
            public final double response;
            public final double[] descriptor;
            
            public Keypoint(double x, double y, double scale, double orientation, double response, double[] descriptor) {
                this.x = x;
                this.y = y;
                this.scale = scale;
                this.orientation = orientation;
                this.response = response;
                this.descriptor = descriptor != null ? descriptor.clone() : null;
            }
            
            public String toString() {
                return String.format("Keypoint[x=%.1f, y=%.1f, scale=%.2f, orientation=%.2f°, response=%.3f]",
                                   x, y, scale, Math.toDegrees(orientation), response);
            }
        }
        
        // SIFT-like feature detector (simplified)
        public static List<Keypoint> detectSIFTFeatures(double[][] image, int octaves, int scales) {
            List<Keypoint> keypoints = new ArrayList<>();
            int height = image.length;
            int width = image[0].length;
            
            // Build scale space
            double[][][] scaleSpace = buildScaleSpace(image, octaves, scales);
            
            // Detect extrema in scale space
            for (int o = 0; o < octaves; o++) {
                for (int s = 1; s < scales - 1; s++) {
                    double[][] current = scaleSpace[o * scales + s];
                    double[][] below = scaleSpace[o * scales + s - 1];
                    double[][] above = scaleSpace[o * scales + s + 1];
                    
                    int currentHeight = current.length;
                    int currentWidth = current[0].length;
                    
                    for (int y = 1; y < currentHeight - 1; y++) {
                        for (int x = 1; x < currentWidth - 1; x++) {
                            if (isLocalExtremum(x, y, current, below, above)) {
                                double scale = Math.pow(2, o) * Math.pow(1.6, s);
                                double orientation = computeOrientation(current, x, y);
                                double response = Math.abs(current[y][x]);
                                
                                // Compute descriptor
                                double[] descriptor = computeSIFTDescriptor(current, x, y, orientation);
                                
                                keypoints.add(new Keypoint(x * Math.pow(2, o), y * Math.pow(2, o), 
                                                         scale, orientation, response, descriptor));
                            }
                        }
                    }
                }
            }
            
            // Sort by response strength
            keypoints.sort((a, b) -> Double.compare(b.response, a.response));
            
            // Keep top features
            return keypoints.subList(0, Math.min(500, keypoints.size()));
        }
        
        private static double[][][] buildScaleSpace(double[][] image, int octaves, int scales) {
            double[][][] scaleSpace = new double[octaves * scales][][];
            double sigma = 1.6;
            
            // Initial image
            scaleSpace[0] = gaussianBlur(image, sigma);
            
            for (int o = 0; o < octaves; o++) {
                double[][] octaveImage = (o == 0) ? image : downsample(scaleSpace[(o-1) * scales]);
                
                for (int s = 0; s < scales; s++) {
                    int index = o * scales + s;
                    double currentSigma = sigma * Math.pow(1.6, s);
                    scaleSpace[index] = gaussianBlur(octaveImage, currentSigma);
                }
            }
            
            return scaleSpace;
        }
        
        private static boolean isLocalExtremum(int x, int y, double[][] current, double[][] below, double[][] above) {
            double centerValue = current[y][x];
            
            // Check 3x3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    
                    if (below[y + dy][x + dx] >= centerValue || above[y + dy][x + dx] >= centerValue) {
                        return false;
                    }
                    if (current[y + dy][x + dx] >= centerValue) {
                        return false;
                    }
                }
            }
            
            return Math.abs(centerValue) > 0.03; // Threshold for significance
        }
        
        private static double computeOrientation(double[][] image, int x, int y) {
            double[] histogram = new double[36]; // 36 bins for 360 degrees
            int height = image.length;
            int width = image[0].length;
            
            // Compute gradients in neighborhood
            for (int dy = -3; dy <= 3; dy++) {
                for (int dx = -3; dx <= 3; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1) {
                        double gradX = image[ny][nx + 1] - image[ny][nx - 1];
                        double gradY = image[ny + 1][nx] - image[ny - 1][nx];
                        
                        double magnitude = Math.sqrt(gradX * gradX + gradY * gradY);
                        double angle = Math.atan2(gradY, gradX);
                        
                        // Gaussian weighting
                        double weight = Math.exp(-(dx * dx + dy * dy) / (2 * 1.5 * 1.5));
                        
                        int bin = (int) (((angle + Math.PI) / (2 * Math.PI)) * 36);
                        bin = Math.max(0, Math.min(35, bin));
                        
                        histogram[bin] += magnitude * weight;
                    }
                }
            }
            
            // Find dominant orientation
            int maxBin = 0;
            for (int i = 1; i < 36; i++) {
                if (histogram[i] > histogram[maxBin]) {
                    maxBin = i;
                }
            }
            
            return (maxBin * 2 * Math.PI / 36) - Math.PI;
        }
        
        private static double[] computeSIFTDescriptor(double[][] image, int x, int y, double orientation) {
            double[] descriptor = new double[128]; // 4x4 grid of 8-bin histograms
            int height = image.length;
            int width = image[0].length;
            
            double cos_theta = Math.cos(-orientation);
            double sin_theta = Math.sin(-orientation);
            
            for (int dy = -8; dy < 8; dy++) {
                for (int dx = -8; dx < 8; dx++) {
                    // Rotate coordinates
                    double rotX = dx * cos_theta - dy * sin_theta;
                    double rotY = dx * sin_theta + dy * cos_theta;
                    
                    int nx = x + (int) rotX;
                    int ny = y + (int) rotY;
                    
                    if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1) {
                        double gradX = image[ny][nx + 1] - image[ny][nx - 1];
                        double gradY = image[ny + 1][nx] - image[ny - 1][nx];
                        
                        double magnitude = Math.sqrt(gradX * gradX + gradY * gradY);
                        double angle = Math.atan2(gradY, gradX) - orientation;
                        
                        // Determine which 4x4 cell and 8-bin histogram
                        int cellX = (int) ((rotX + 8) / 4);
                        int cellY = (int) ((rotY + 8) / 4);
                        
                        if (cellX >= 0 && cellX < 4 && cellY >= 0 && cellY < 4) {
                            int bin = (int) (((angle + Math.PI) / (2 * Math.PI)) * 8);
                            bin = Math.max(0, Math.min(7, bin));
                            
                            int index = cellY * 32 + cellX * 8 + bin;
                            descriptor[index] += magnitude;
                        }
                    }
                }
            }
            
            // Normalize descriptor
            double norm = 0;
            for (double d : descriptor) {
                norm += d * d;
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < descriptor.length; i++) {
                    descriptor[i] /= norm;
                    descriptor[i] = Math.min(descriptor[i], 0.2); // Clamp to reduce illumination effects
                }
                
                // Renormalize
                norm = 0;
                for (double d : descriptor) {
                    norm += d * d;
                }
                norm = Math.sqrt(norm);
                
                if (norm > 0) {
                    for (int i = 0; i < descriptor.length; i++) {
                        descriptor[i] /= norm;
                    }
                }
            }
            
            return descriptor;
        }
        
        // ORB-like feature detector (simplified)
        public static List<Keypoint> detectORBFeatures(double[][] image, int maxFeatures) {
            List<Keypoint> keypoints = new ArrayList<>();
            
            // Detect FAST corners
            List<Keypoint> corners = detectFASTCorners(image, 0.2);
            
            // Sort by response and take top features
            corners.sort((a, b) -> Double.compare(b.response, a.response));
            corners = corners.subList(0, Math.min(maxFeatures, corners.size()));
            
            // Compute BRIEF descriptors
            for (Keypoint corner : corners) {
                double[] descriptor = computeBRIEFDescriptor(image, (int) corner.x, (int) corner.y);
                keypoints.add(new Keypoint(corner.x, corner.y, corner.scale, corner.orientation, 
                                         corner.response, descriptor));
            }
            
            return keypoints;
        }
        
        private static List<Keypoint> detectFASTCorners(double[][] image, double threshold) {
            List<Keypoint> corners = new ArrayList<>();
            int height = image.length;
            int width = image[0].length;
            
            // FAST-9 circle offsets
            int[][] circle = {
                {3, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 3}, {-1, 3}, {-2, 2}, {-3, 1},
                {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}, {0, -3}, {1, -3}, {2, -2}, {3, -1}
            };
            
            for (int y = 3; y < height - 3; y++) {
                for (int x = 3; x < width - 3; x++) {
                    double centerValue = image[y][x];
                    
                    int brightCount = 0;
                    int darkCount = 0;
                    
                    for (int[] offset : circle) {
                        double pixelValue = image[y + offset[1]][x + offset[0]];
                        
                        if (pixelValue > centerValue + threshold) {
                            brightCount++;
                        } else if (pixelValue < centerValue - threshold) {
                            darkCount++;
                        }
                    }
                    
                    if (brightCount >= 9 || darkCount >= 9) {
                        double response = Math.max(brightCount, darkCount) / 16.0;
                        corners.add(new Keypoint(x, y, 1.0, 0.0, response, null));
                    }
                }
            }
            
            return corners;
        }
        
        private static double[] computeBRIEFDescriptor(double[][] image, int x, int y) {
            Random random = new Random(42); // Fixed seed for reproducibility
            int descriptorLength = 256; // 256 bit descriptor
            double[] descriptor = new double[descriptorLength];
            
            int height = image.length;
            int width = image[0].length;
            int patchSize = 15;
            
            for (int i = 0; i < descriptorLength; i++) {
                // Generate random point pairs
                int x1 = x + random.nextInt(patchSize) - patchSize / 2;
                int y1 = y + random.nextInt(patchSize) - patchSize / 2;
                int x2 = x + random.nextInt(patchSize) - patchSize / 2;
                int y2 = y + random.nextInt(patchSize) - patchSize / 2;
                
                // Clamp to image bounds
                x1 = Math.max(0, Math.min(width - 1, x1));
                y1 = Math.max(0, Math.min(height - 1, y1));
                x2 = Math.max(0, Math.min(width - 1, x2));
                y2 = Math.max(0, Math.min(height - 1, y2));
                
                descriptor[i] = (image[y1][x1] < image[y2][x2]) ? 1.0 : 0.0;
            }
            
            return descriptor;
        }
        
        private static double[][] gaussianBlur(double[][] image, double sigma) {
            int height = image.length;
            int width = image[0].length;
            
            // Create Gaussian kernel
            int kernelSize = (int) (6 * sigma + 1);
            if (kernelSize % 2 == 0) kernelSize++;
            int radius = kernelSize / 2;
            
            double[] kernel = new double[kernelSize];
            double sum = 0;
            
            for (int i = 0; i < kernelSize; i++) {
                int x = i - radius;
                kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
                sum += kernel[i];
            }
            
            // Normalize kernel
            for (int i = 0; i < kernelSize; i++) {
                kernel[i] /= sum;
            }
            
            // Apply separable convolution
            double[][] temp = new double[height][width];
            double[][] result = new double[height][width];
            
            // Horizontal pass
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = 0;
                    for (int k = 0; k < kernelSize; k++) {
                        int nx = x + k - radius;
                        nx = Math.max(0, Math.min(width - 1, nx));
                        value += image[y][nx] * kernel[k];
                    }
                    temp[y][x] = value;
                }
            }
            
            // Vertical pass
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = 0;
                    for (int k = 0; k < kernelSize; k++) {
                        int ny = y + k - radius;
                        ny = Math.max(0, Math.min(height - 1, ny));
                        value += temp[ny][x] * kernel[k];
                    }
                    result[y][x] = value;
                }
            }
            
            return result;
        }
        
        private static double[][] downsample(double[][] image) {
            int height = image.length;
            int width = image[0].length;
            int newHeight = height / 2;
            int newWidth = width / 2;
            
            double[][] result = new double[newHeight][newWidth];
            
            for (int y = 0; y < newHeight; y++) {
                for (int x = 0; x < newWidth; x++) {
                    result[y][x] = image[y * 2][x * 2];
                }
            }
            
            return result;
        }
    }
    
    /**
     * Feature Matching Algorithms
     */
    public static class FeatureMatching {
        
        public static class Match {
            public final FeatureDetection.Keypoint keypoint1;
            public final FeatureDetection.Keypoint keypoint2;
            public final double distance;
            public final double confidence;
            
            public Match(FeatureDetection.Keypoint kp1, FeatureDetection.Keypoint kp2, double distance, double confidence) {
                this.keypoint1 = kp1;
                this.keypoint2 = kp2;
                this.distance = distance;
                this.confidence = confidence;
            }
            
            public String toString() {
                return String.format("Match[distance=%.3f, confidence=%.3f]", distance, confidence);
            }
        }
        
        public static List<Match> bruteForceMatching(List<FeatureDetection.Keypoint> features1,
                                                   List<FeatureDetection.Keypoint> features2,
                                                   double maxDistance) {
            List<Match> matches = new ArrayList<>();
            
            for (FeatureDetection.Keypoint kp1 : features1) {
                if (kp1.descriptor == null) continue;
                
                double bestDistance = Double.MAX_VALUE;
                double secondBestDistance = Double.MAX_VALUE;
                FeatureDetection.Keypoint bestMatch = null;
                
                for (FeatureDetection.Keypoint kp2 : features2) {
                    if (kp2.descriptor == null) continue;
                    
                    double distance = computeDescriptorDistance(kp1.descriptor, kp2.descriptor);
                    
                    if (distance < bestDistance) {
                        secondBestDistance = bestDistance;
                        bestDistance = distance;
                        bestMatch = kp2;
                    } else if (distance < secondBestDistance) {
                        secondBestDistance = distance;
                    }
                }
                
                // Lowe's ratio test
                if (bestMatch != null && bestDistance < maxDistance && 
                    bestDistance / secondBestDistance < 0.8) {
                    double confidence = 1.0 - (bestDistance / secondBestDistance);
                    matches.add(new Match(kp1, bestMatch, bestDistance, confidence));
                }
            }
            
            return matches;
        }
        
        public static List<Match> flannMatching(List<FeatureDetection.Keypoint> features1,
                                              List<FeatureDetection.Keypoint> features2,
                                              double maxDistance) {
            // Simplified FLANN-like matching using KD-tree approximation
            return bruteForceMatching(features1, features2, maxDistance);
        }
        
        private static double computeDescriptorDistance(double[] desc1, double[] desc2) {
            if (desc1.length != desc2.length) {
                throw new IllegalArgumentException("Descriptor lengths must match");
            }
            
            // For binary descriptors (like BRIEF), use Hamming distance
            if (isBinaryDescriptor(desc1)) {
                int hammingDistance = 0;
                for (int i = 0; i < desc1.length; i++) {
                    if (desc1[i] != desc2[i]) {
                        hammingDistance++;
                    }
                }
                return hammingDistance;
            } else {
                // For real-valued descriptors (like SIFT), use Euclidean distance
                double sum = 0;
                for (int i = 0; i < desc1.length; i++) {
                    double diff = desc1[i] - desc2[i];
                    sum += diff * diff;
                }
                return Math.sqrt(sum);
            }
        }
        
        private static boolean isBinaryDescriptor(double[] descriptor) {
            for (double value : descriptor) {
                if (value != 0.0 && value != 1.0) {
                    return false;
                }
            }
            return true;
        }
    }
    
    /**
     * Geometric Transformation and RANSAC
     */
    public static class GeometricTransformation {
        
        public static class Homography {
            public final double[][] matrix; // 3x3 homography matrix
            
            public Homography(double[][] matrix) {
                this.matrix = new double[3][3];
                for (int i = 0; i < 3; i++) {
                    System.arraycopy(matrix[i], 0, this.matrix[i], 0, 3);
                }
            }
            
            public double[] transform(double x, double y) {
                double w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
                if (Math.abs(w) < 1e-8) w = 1e-8;
                
                return new double[]{
                    (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
                    (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w
                };
            }
            
            public String toString() {
                StringBuilder sb = new StringBuilder("Homography:\n");
                for (int i = 0; i < 3; i++) {
                    sb.append(String.format("[%8.4f %8.4f %8.4f]\n", matrix[i][0], matrix[i][1], matrix[i][2]));
                }
                return sb.toString();
            }
        }
        
        public static class RANSACResult {
            public final Homography homography;
            public final List<FeatureMatching.Match> inliers;
            public final double inlierRatio;
            
            public RANSACResult(Homography homography, List<FeatureMatching.Match> inliers, double inlierRatio) {
                this.homography = homography;
                this.inliers = new ArrayList<>(inliers);
                this.inlierRatio = inlierRatio;
            }
            
            public String toString() {
                return String.format("RANSAC[inliers=%d, ratio=%.3f]", inliers.size(), inlierRatio);
            }
        }
        
        public static RANSACResult findHomographyRANSAC(List<FeatureMatching.Match> matches,
                                                       double threshold, int maxIterations) {
            if (matches.size() < 4) {
                throw new IllegalArgumentException("At least 4 matches required for homography estimation");
            }
            
            Random random = new Random();
            int bestInlierCount = 0;
            Homography bestHomography = null;
            List<FeatureMatching.Match> bestInliers = new ArrayList<>();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                // Randomly select 4 matches
                List<FeatureMatching.Match> sample = new ArrayList<>();
                Set<Integer> selectedIndices = new HashSet<>();
                
                while (sample.size() < 4) {
                    int index = random.nextInt(matches.size());
                    if (!selectedIndices.contains(index)) {
                        selectedIndices.add(index);
                        sample.add(matches.get(index));
                    }
                }
                
                // Compute homography from 4 points
                try {
                    Homography homography = computeHomography(sample);
                    
                    // Count inliers
                    List<FeatureMatching.Match> inliers = new ArrayList<>();
                    for (FeatureMatching.Match match : matches) {
                        double error = computeReprojectionError(homography, match);
                        if (error < threshold) {
                            inliers.add(match);
                        }
                    }
                    
                    if (inliers.size() > bestInlierCount) {
                        bestInlierCount = inliers.size();
                        bestHomography = homography;
                        bestInliers = new ArrayList<>(inliers);
                    }
                } catch (Exception e) {
                    // Skip this iteration if homography computation fails
                    continue;
                }
            }
            
            if (bestHomography != null && bestInliers.size() >= 4) {
                // Refine using all inliers
                bestHomography = computeHomography(bestInliers);
                double inlierRatio = (double) bestInliers.size() / matches.size();
                return new RANSACResult(bestHomography, bestInliers, inlierRatio);
            }
            
            return null;
        }
        
        private static Homography computeHomography(List<FeatureMatching.Match> matches) {
            if (matches.size() < 4) {
                throw new IllegalArgumentException("At least 4 point correspondences required");
            }
            
            // Set up linear system Ah = 0 using DLT (Direct Linear Transform)
            int n = matches.size();
            double[][] A = new double[2 * n][9];
            
            for (int i = 0; i < n; i++) {
                FeatureMatching.Match match = matches.get(i);
                double x1 = match.keypoint1.x;
                double y1 = match.keypoint1.y;
                double x2 = match.keypoint2.x;
                double y2 = match.keypoint2.y;
                
                // First equation: x2(u1*h31 + v1*h32 + h33) = u1*h11 + v1*h12 + h13
                A[2*i][0] = x1;
                A[2*i][1] = y1;
                A[2*i][2] = 1;
                A[2*i][3] = 0;
                A[2*i][4] = 0;
                A[2*i][5] = 0;
                A[2*i][6] = -x2 * x1;
                A[2*i][7] = -x2 * y1;
                A[2*i][8] = -x2;
                
                // Second equation: y2(u1*h31 + v1*h32 + h33) = u1*h21 + v1*h22 + h23
                A[2*i+1][0] = 0;
                A[2*i+1][1] = 0;
                A[2*i+1][2] = 0;
                A[2*i+1][3] = x1;
                A[2*i+1][4] = y1;
                A[2*i+1][5] = 1;
                A[2*i+1][6] = -y2 * x1;
                A[2*i+1][7] = -y2 * y1;
                A[2*i+1][8] = -y2;
            }
            
            // Solve using SVD (simplified version)
            double[] h = solveDLT(A);
            
            // Reshape solution vector to 3x3 matrix
            double[][] homography = new double[3][3];
            homography[0][0] = h[0]; homography[0][1] = h[1]; homography[0][2] = h[2];
            homography[1][0] = h[3]; homography[1][1] = h[4]; homography[1][2] = h[5];
            homography[2][0] = h[6]; homography[2][1] = h[7]; homography[2][2] = h[8];
            
            return new Homography(homography);
        }
        
        private static double[] solveDLT(double[][] A) {
            // Simplified SVD solver - in practice, use a proper SVD implementation
            // For demonstration, use least squares approximation
            
            int rows = A.length;
            int cols = A[0].length;
            
            // Compute A^T * A
            double[][] ATA = new double[cols][cols];
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < cols; j++) {
                    double sum = 0;
                    for (int k = 0; k < rows; k++) {
                        sum += A[k][i] * A[k][j];
                    }
                    ATA[i][j] = sum;
                }
            }
            
            // Find eigenvector corresponding to smallest eigenvalue
            // Simplified: use power iteration on (I - ATA/||ATA||)
            double[] h = new double[cols];
            Random random = new Random(42);
            for (int i = 0; i < cols; i++) {
                h[i] = random.nextGaussian();
            }
            
            // Normalize ATA
            double norm = 0;
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < cols; j++) {
                    norm += ATA[i][j] * ATA[i][j];
                }
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < cols; i++) {
                    for (int j = 0; j < cols; j++) {
                        ATA[i][j] /= norm;
                    }
                }
            }
            
            // Power iteration (inverse)
            for (int iter = 0; iter < 100; iter++) {
                double[] newH = new double[cols];
                
                for (int i = 0; i < cols; i++) {
                    newH[i] = h[i];
                    for (int j = 0; j < cols; j++) {
                        newH[i] -= ATA[i][j] * h[j];
                    }
                }
                
                // Normalize
                double hNorm = 0;
                for (double val : newH) {
                    hNorm += val * val;
                }
                hNorm = Math.sqrt(hNorm);
                
                if (hNorm > 0) {
                    for (int i = 0; i < cols; i++) {
                        newH[i] /= hNorm;
                    }
                }
                
                h = newH;
            }
            
            return h;
        }
        
        private static double computeReprojectionError(Homography homography, FeatureMatching.Match match) {
            double[] projected = homography.transform(match.keypoint1.x, match.keypoint1.y);
            double dx = projected[0] - match.keypoint2.x;
            double dy = projected[1] - match.keypoint2.y;
            return Math.sqrt(dx * dx + dy * dy);
        }
    }
    
    /**
     * Object Detection using Template Matching and Cascade Classifiers
     */
    public static class ObjectDetection {
        
        public static class Detection {
            public final int x, y, width, height;
            public final double confidence;
            public final String className;
            
            public Detection(int x, int y, int width, int height, double confidence, String className) {
                this.x = x;
                this.y = y;
                this.width = width;
                this.height = height;
                this.confidence = confidence;
                this.className = className;
            }
            
            public String toString() {
                return String.format("Detection[%s at (%d,%d) %dx%d, conf=%.3f]", 
                                   className, x, y, width, height, confidence);
            }
        }
        
        public static class Template {
            public final double[][] image;
            public final String className;
            
            public Template(double[][] image, String className) {
                this.image = image;
                this.className = className;
            }
        }
        
        public static List<Detection> templateMatching(double[][] image, Template template, double threshold) {
            List<Detection> detections = new ArrayList<>();
            int imageHeight = image.length;
            int imageWidth = image[0].length;
            int templateHeight = template.image.length;
            int templateWidth = template.image[0].length;
            
            if (templateHeight >= imageHeight || templateWidth >= imageWidth) {
                return detections;
            }
            
            // Compute normalized cross-correlation
            for (int y = 0; y <= imageHeight - templateHeight; y++) {
                for (int x = 0; x <= imageWidth - templateWidth; x++) {
                    double correlation = computeNormalizedCrossCorrelation(image, template.image, x, y);
                    
                    if (correlation > threshold) {
                        detections.add(new Detection(x, y, templateWidth, templateHeight, 
                                                   correlation, template.className));
                    }
                }
            }
            
            // Non-maximum suppression
            return nonMaximumSuppression(detections, 0.3);
        }
        
        private static double computeNormalizedCrossCorrelation(double[][] image, double[][] template, int startX, int startY) {
            int templateHeight = template.length;
            int templateWidth = template[0].length;
            
            // Compute means
            double imageMean = 0;
            double templateMean = 0;
            
            for (int y = 0; y < templateHeight; y++) {
                for (int x = 0; x < templateWidth; x++) {
                    imageMean += image[startY + y][startX + x];
                    templateMean += template[y][x];
                }
            }
            
            int numPixels = templateHeight * templateWidth;
            imageMean /= numPixels;
            templateMean /= numPixels;
            
            // Compute correlation
            double numerator = 0;
            double imageVariance = 0;
            double templateVariance = 0;
            
            for (int y = 0; y < templateHeight; y++) {
                for (int x = 0; x < templateWidth; x++) {
                    double imageDiff = image[startY + y][startX + x] - imageMean;
                    double templateDiff = template[y][x] - templateMean;
                    
                    numerator += imageDiff * templateDiff;
                    imageVariance += imageDiff * imageDiff;
                    templateVariance += templateDiff * templateDiff;
                }
            }
            
            double denominator = Math.sqrt(imageVariance * templateVariance);
            return (denominator > 0) ? numerator / denominator : 0;
        }
        
        private static List<Detection> nonMaximumSuppression(List<Detection> detections, double iouThreshold) {
            // Sort by confidence
            detections.sort((a, b) -> Double.compare(b.confidence, a.confidence));
            
            List<Detection> result = new ArrayList<>();
            boolean[] suppressed = new boolean[detections.size()];
            
            for (int i = 0; i < detections.size(); i++) {
                if (suppressed[i]) continue;
                
                Detection det1 = detections.get(i);
                result.add(det1);
                
                for (int j = i + 1; j < detections.size(); j++) {
                    if (suppressed[j]) continue;
                    
                    Detection det2 = detections.get(j);
                    double iou = computeIoU(det1, det2);
                    
                    if (iou > iouThreshold) {
                        suppressed[j] = true;
                    }
                }
            }
            
            return result;
        }
        
        private static double computeIoU(Detection det1, Detection det2) {
            int x1 = Math.max(det1.x, det2.x);
            int y1 = Math.max(det1.y, det2.y);
            int x2 = Math.min(det1.x + det1.width, det2.x + det2.width);
            int y2 = Math.min(det1.y + det1.height, det2.y + det2.height);
            
            if (x2 <= x1 || y2 <= y1) return 0.0;
            
            double intersection = (x2 - x1) * (y2 - y1);
            double area1 = det1.width * det1.height;
            double area2 = det2.width * det2.height;
            double union = area1 + area2 - intersection;
            
            return intersection / union;
        }
        
        // Simplified Haar-like cascade classifier
        public static class HaarCascade {
            private final List<HaarFeature> features;
            private final double[] thresholds;
            private final String className;
            
            public HaarCascade(String className) {
                this.className = className;
                this.features = generateHaarFeatures();
                this.thresholds = new double[features.size()];
                
                // Initialize with default thresholds
                for (int i = 0; i < thresholds.length; i++) {
                    thresholds[i] = 0.5;
                }
            }
            
            public List<Detection> detect(double[][] image, int windowSize) {
                List<Detection> detections = new ArrayList<>();
                int height = image.length;
                int width = image[0].length;
                
                // Compute integral image
                double[][] integralImage = computeIntegralImage(image);
                
                // Slide window
                for (int y = 0; y <= height - windowSize; y += 4) {
                    for (int x = 0; x <= width - windowSize; x += 4) {
                        double confidence = evaluateWindow(integralImage, x, y, windowSize);
                        
                        if (confidence > 0.7) {
                            detections.add(new Detection(x, y, windowSize, windowSize, confidence, className));
                        }
                    }
                }
                
                return nonMaximumSuppression(detections, 0.3);
            }
            
            private double evaluateWindow(double[][] integralImage, int x, int y, int size) {
                double score = 0;
                int passedStages = 0;
                
                for (int i = 0; i < features.size(); i++) {
                    double featureValue = features.get(i).evaluate(integralImage, x, y, size);
                    
                    if (featureValue > thresholds[i]) {
                        score += featureValue;
                        passedStages++;
                    } else {
                        break; // Cascade failed
                    }
                }
                
                return (double) passedStages / features.size();
            }
            
            private List<HaarFeature> generateHaarFeatures() {
                List<HaarFeature> features = new ArrayList<>();
                
                // Add different types of Haar features
                features.add(new HaarFeature(HaarFeature.Type.EDGE_HORIZONTAL, 0.25, 0.25, 0.5, 0.5));
                features.add(new HaarFeature(HaarFeature.Type.EDGE_VERTICAL, 0.25, 0.25, 0.5, 0.5));
                features.add(new HaarFeature(HaarFeature.Type.LINE_HORIZONTAL, 0.2, 0.2, 0.6, 0.6));
                features.add(new HaarFeature(HaarFeature.Type.LINE_VERTICAL, 0.2, 0.2, 0.6, 0.6));
                features.add(new HaarFeature(HaarFeature.Type.FOUR_RECTANGLE, 0.1, 0.1, 0.8, 0.8));
                
                return features;
            }
        }
        
        public static class HaarFeature {
            public enum Type {
                EDGE_HORIZONTAL, EDGE_VERTICAL, LINE_HORIZONTAL, LINE_VERTICAL, FOUR_RECTANGLE
            }
            
            private final Type type;
            private final double x, y, width, height;
            
            public HaarFeature(Type type, double x, double y, double width, double height) {
                this.type = type;
                this.x = x;
                this.y = y;
                this.width = width;
                this.height = height;
            }
            
            public double evaluate(double[][] integralImage, int windowX, int windowY, int windowSize) {
                int featureX = (int) (windowX + x * windowSize);
                int featureY = (int) (windowY + y * windowSize);
                int featureWidth = (int) (width * windowSize);
                int featureHeight = (int) (height * windowSize);
                
                switch (type) {
                    case EDGE_HORIZONTAL:
                        return evaluateEdgeHorizontal(integralImage, featureX, featureY, featureWidth, featureHeight);
                    case EDGE_VERTICAL:
                        return evaluateEdgeVertical(integralImage, featureX, featureY, featureWidth, featureHeight);
                    case LINE_HORIZONTAL:
                        return evaluateLineHorizontal(integralImage, featureX, featureY, featureWidth, featureHeight);
                    case LINE_VERTICAL:
                        return evaluateLineVertical(integralImage, featureX, featureY, featureWidth, featureHeight);
                    case FOUR_RECTANGLE:
                        return evaluateFourRectangle(integralImage, featureX, featureY, featureWidth, featureHeight);
                    default:
                        return 0;
                }
            }
            
            private double evaluateEdgeHorizontal(double[][] integral, int x, int y, int w, int h) {
                double upper = rectangleSum(integral, x, y, w, h/2);
                double lower = rectangleSum(integral, x, y + h/2, w, h/2);
                return Math.abs(upper - lower) / (w * h);
            }
            
            private double evaluateEdgeVertical(double[][] integral, int x, int y, int w, int h) {
                double left = rectangleSum(integral, x, y, w/2, h);
                double right = rectangleSum(integral, x + w/2, y, w/2, h);
                return Math.abs(left - right) / (w * h);
            }
            
            private double evaluateLineHorizontal(double[][] integral, int x, int y, int w, int h) {
                double upper = rectangleSum(integral, x, y, w, h/3);
                double middle = rectangleSum(integral, x, y + h/3, w, h/3);
                double lower = rectangleSum(integral, x, y + 2*h/3, w, h/3);
                return Math.abs(2 * middle - upper - lower) / (w * h);
            }
            
            private double evaluateLineVertical(double[][] integral, int x, int y, int w, int h) {
                double left = rectangleSum(integral, x, y, w/3, h);
                double middle = rectangleSum(integral, x + w/3, y, w/3, h);
                double right = rectangleSum(integral, x + 2*w/3, y, w/3, h);
                return Math.abs(2 * middle - left - right) / (w * h);
            }
            
            private double evaluateFourRectangle(double[][] integral, int x, int y, int w, int h) {
                double topLeft = rectangleSum(integral, x, y, w/2, h/2);
                double topRight = rectangleSum(integral, x + w/2, y, w/2, h/2);
                double bottomLeft = rectangleSum(integral, x, y + h/2, w/2, h/2);
                double bottomRight = rectangleSum(integral, x + w/2, y + h/2, w/2, h/2);
                return Math.abs((topLeft + bottomRight) - (topRight + bottomLeft)) / (w * h);
            }
            
            private double rectangleSum(double[][] integral, int x, int y, int w, int h) {
                if (x < 0 || y < 0 || x + w >= integral[0].length || y + h >= integral.length) {
                    return 0;
                }
                
                double sum = integral[y + h][x + w];
                if (x > 0) sum -= integral[y + h][x];
                if (y > 0) sum -= integral[y][x + w];
                if (x > 0 && y > 0) sum += integral[y][x];
                
                return sum;
            }
        }
        
        private static double[][] computeIntegralImage(double[][] image) {
            int height = image.length;
            int width = image[0].length;
            double[][] integral = new double[height + 1][width + 1];
            
            for (int y = 1; y <= height; y++) {
                for (int x = 1; x <= width; x++) {
                    integral[y][x] = image[y-1][x-1] + integral[y-1][x] + integral[y][x-1] - integral[y-1][x-1];
                }
            }
            
            return integral;
        }
    }
    
    /**
     * Stereo Vision and 3D Reconstruction
     */
    public static class StereoVision {
        
        public static class Point3D {
            public final double x, y, z;
            
            public Point3D(double x, double y, z) {
                this.x = x;
                this.y = y;
                this.z = z;
            }
            
            public String toString() {
                return String.format("Point3D[%.3f, %.3f, %.3f]", x, y, z);
            }
        }
        
        public static class StereoCamera {
            public final double focal; // focal length
            public final double baseline; // distance between cameras
            public final int imageWidth, imageHeight;
            
            public StereoCamera(double focal, double baseline, int imageWidth, int imageHeight) {
                this.focal = focal;
                this.baseline = baseline;
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
            }
        }
        
        public static double[][] computeDisparityMap(double[][] leftImage, double[][] rightImage, 
                                                   int windowSize, int maxDisparity) {
            int height = leftImage.length;
            int width = leftImage[0].length;
            double[][] disparityMap = new double[height][width];
            
            int halfWindow = windowSize / 2;
            
            for (int y = halfWindow; y < height - halfWindow; y++) {
                for (int x = halfWindow; x < width - halfWindow; x++) {
                    double bestDisparity = 0;
                    double bestCost = Double.MAX_VALUE;
                    
                    // Search for best disparity
                    for (int d = 0; d <= maxDisparity && x - d >= halfWindow; d++) {
                        double cost = computeMatchingCost(leftImage, rightImage, x, y, x - d, y, windowSize);
                        
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestDisparity = d;
                        }
                    }
                    
                    disparityMap[y][x] = bestDisparity;
                }
            }
            
            return disparityMap;
        }
        
        private static double computeMatchingCost(double[][] leftImage, double[][] rightImage,
                                                int leftX, int leftY, int rightX, int rightY, int windowSize) {
            int halfWindow = windowSize / 2;
            double cost = 0;
            int count = 0;
            
            for (int dy = -halfWindow; dy <= halfWindow; dy++) {
                for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                    int ly = leftY + dy;
                    int lx = leftX + dx;
                    int ry = rightY + dy;
                    int rx = rightX + dx;
                    
                    if (ly >= 0 && ly < leftImage.length && lx >= 0 && lx < leftImage[0].length &&
                        ry >= 0 && ry < rightImage.length && rx >= 0 && rx < rightImage[0].length) {
                        
                        double diff = leftImage[ly][lx] - rightImage[ry][rx];
                        cost += diff * diff; // SSD (Sum of Squared Differences)
                        count++;
                    }
                }
            }
            
            return count > 0 ? cost / count : Double.MAX_VALUE;
        }
        
        public static List<Point3D> reconstructPointCloud(double[][] disparityMap, StereoCamera camera) {
            List<Point3D> pointCloud = new ArrayList<>();
            int height = disparityMap.length;
            int width = disparityMap[0].length;
            
            double cx = width / 2.0;  // Principal point x
            double cy = height / 2.0; // Principal point y
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double disparity = disparityMap[y][x];
                    
                    if (disparity > 0) {
                        // Compute 3D coordinates
                        double z = (camera.focal * camera.baseline) / disparity;
                        double worldX = (x - cx) * z / camera.focal;
                        double worldY = (y - cy) * z / camera.focal;
                        
                        pointCloud.add(new Point3D(worldX, worldY, z));
                    }
                }
            }
            
            return pointCloud;
        }
        
        public static double[][] computeDepthMap(double[][] disparityMap, StereoCamera camera) {
            int height = disparityMap.length;
            int width = disparityMap[0].length;
            double[][] depthMap = new double[height][width];
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double disparity = disparityMap[y][x];
                    if (disparity > 0) {
                        depthMap[y][x] = (camera.focal * camera.baseline) / disparity;
                    } else {
                        depthMap[y][x] = Double.POSITIVE_INFINITY;
                    }
                }
            }
            
            return depthMap;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Computer Vision Algorithms Demo:");
        System.out.println("===============================");
        
        // Create sample images
        double[][] image1 = createSampleImage(100, 100, "checkerboard");
        double[][] image2 = createSampleImage(100, 100, "circles");
        
        // Feature Detection
        System.out.println("1. Feature Detection:");
        List<FeatureDetection.Keypoint> siftFeatures = FeatureDetection.detectSIFTFeatures(image1, 3, 5);
        System.out.printf("SIFT features detected: %d\n", siftFeatures.size());
        if (!siftFeatures.isEmpty()) {
            System.out.println("Top SIFT feature: " + siftFeatures.get(0));
        }
        
        List<FeatureDetection.Keypoint> orbFeatures = FeatureDetection.detectORBFeatures(image1, 100);
        System.out.printf("ORB features detected: %d\n", orbFeatures.size());
        if (!orbFeatures.isEmpty()) {
            System.out.println("Top ORB feature: " + orbFeatures.get(0));
        }
        
        // Feature Matching
        System.out.println("\n2. Feature Matching:");
        if (siftFeatures.size() > 0 && orbFeatures.size() > 0) {
            List<FeatureMatching.Match> matches = FeatureMatching.bruteForceMatching(
                siftFeatures.subList(0, Math.min(10, siftFeatures.size())),
                orbFeatures.subList(0, Math.min(10, orbFeatures.size())),
                1.0);
            System.out.printf("Feature matches found: %d\n", matches.size());
            if (!matches.isEmpty()) {
                System.out.println("Best match: " + matches.get(0));
            }
        }
        
        // Geometric Transformation
        System.out.println("\n3. Geometric Transformation:");
        if (siftFeatures.size() >= 4) {
            // Create artificial matches for demonstration
            List<FeatureMatching.Match> artificialMatches = new ArrayList<>();
            for (int i = 0; i < Math.min(4, siftFeatures.size()); i++) {
                FeatureDetection.Keypoint kp1 = siftFeatures.get(i);
                // Create a slightly translated keypoint as match
                FeatureDetection.Keypoint kp2 = new FeatureDetection.Keypoint(
                    kp1.x + 5, kp1.y + 3, kp1.scale, kp1.orientation, kp1.response, kp1.descriptor);
                artificialMatches.add(new FeatureMatching.Match(kp1, kp2, 0.1, 0.9));
            }
            
            GeometricTransformation.RANSACResult ransacResult = 
                GeometricTransformation.findHomographyRANSAC(artificialMatches, 2.0, 1000);
            
            if (ransacResult != null) {
                System.out.println("RANSAC result: " + ransacResult);
                System.out.println(ransacResult.homography);
            }
        }
        
        // Object Detection
        System.out.println("4. Object Detection:");
        ObjectDetection.Template template = new ObjectDetection.Template(
            createSampleImage(20, 20, "cross"), "cross");
        
        List<ObjectDetection.Detection> detections = 
            ObjectDetection.templateMatching(image1, template, 0.7);
        System.out.printf("Template matching detections: %d\n", detections.size());
        
        // Haar Cascade
        ObjectDetection.HaarCascade cascade = new ObjectDetection.HaarCascade("object");
        List<ObjectDetection.Detection> cascadeDetections = cascade.detect(image1, 24);
        System.out.printf("Haar cascade detections: %d\n", cascadeDetections.size());
        
        // Stereo Vision
        System.out.println("\n5. Stereo Vision:");
        StereoVision.StereoCamera camera = new StereoVision.StereoCamera(500, 0.1, 100, 100);
        
        double[][] leftImage = createSampleImage(100, 100, "gradient");
        double[][] rightImage = createSampleImage(100, 100, "gradient_shifted");
        
        double[][] disparityMap = StereoVision.computeDisparityMap(leftImage, rightImage, 5, 20);
        List<StereoVision.Point3D> pointCloud = StereoVision.reconstructPointCloud(disparityMap, camera);
        
        System.out.printf("Point cloud size: %d points\n", pointCloud.size());
        if (!pointCloud.isEmpty()) {
            System.out.println("Sample 3D point: " + pointCloud.get(0));
        }
        
        double[][] depthMap = StereoVision.computeDepthMap(disparityMap, camera);
        double avgDepth = Arrays.stream(depthMap)
            .flatMapToDouble(Arrays::stream)
            .filter(d -> !Double.isInfinite(d))
            .average()
            .orElse(0);
        System.out.printf("Average depth: %.3f\n", avgDepth);
        
        System.out.println("\nComputer vision demonstration completed!");
        System.out.println("Algorithms demonstrated:");
        System.out.println("- Feature detection: SIFT, ORB, FAST");
        System.out.println("- Feature matching: Brute force, FLANN");
        System.out.println("- Geometric transformation: Homography estimation with RANSAC");
        System.out.println("- Object detection: Template matching, Haar cascades");
        System.out.println("- Stereo vision: Disparity computation, 3D reconstruction");
    }
    
    private static double[][] createSampleImage(int width, int height, String pattern) {
        double[][] image = new double[height][width];
        Random random = new Random(42);
        
        switch (pattern) {
            case "checkerboard":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        image[y][x] = ((x / 10) + (y / 10)) % 2 == 0 ? 1.0 : 0.0;
                    }
                }
                break;
                
            case "circles":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double distance = Math.sqrt((x - width/2.0) * (x - width/2.0) + 
                                                  (y - height/2.0) * (y - height/2.0));
                        image[y][x] = Math.sin(distance / 5.0) * 0.5 + 0.5;
                    }
                }
                break;
                
            case "cross":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        image[y][x] = (Math.abs(x - width/2) < 2 || Math.abs(y - height/2) < 2) ? 1.0 : 0.0;
                    }
                }
                break;
                
            case "gradient":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        image[y][x] = (double) x / width;
                    }
                }
                break;
                
            case "gradient_shifted":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int shiftedX = Math.max(0, x - 5); // Simulate camera displacement
                        image[y][x] = (double) shiftedX / width;
                    }
                }
                break;
                
            default:
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        image[y][x] = random.nextDouble();
                    }
                }
        }
        
        return image;
    }
}
