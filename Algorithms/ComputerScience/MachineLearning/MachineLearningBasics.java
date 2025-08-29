package Algorithms.ComputerScience.MachineLearning;

import java.util.*;

/**
 * Basic Machine Learning Algorithms
 */
public class MachineLearningBasics {
    
    /**
     * Linear Regression using Least Squares Method
     */
    public static class LinearRegression {
        private double slope;
        private double intercept;
        
        /**
         * Train the linear regression model
         * @param x Input features
         * @param y Target values
         */
        public void fit(double[] x, double[] y) {
            if (x.length != y.length) {
                throw new IllegalArgumentException("Arrays must have same length");
            }
            
            int n = x.length;
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            
            for (int i = 0; i < n; i++) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumXX += x[i] * x[i];
            }
            
            // Calculate slope and intercept
            slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
            intercept = (sumY - slope * sumX) / n;
        }
        
        /**
         * Make prediction
         * @param x Input value
         * @return Predicted value
         */
        public double predict(double x) {
            return slope * x + intercept;
        }
        
        public double getSlope() { return slope; }
        public double getIntercept() { return intercept; }
    }
    
    /**
     * K-Means Clustering Algorithm
     */
    public static class KMeans {
        private double[][] centroids;
        private int k;
        private int maxIterations;
        
        public KMeans(int k, int maxIterations) {
            this.k = k;
            this.maxIterations = maxIterations;
        }
        
        /**
         * Fit K-means clustering
         * @param data 2D array where each row is a data point
         * @return Cluster assignments for each data point
         */
        public int[] fit(double[][] data) {
            int n = data.length;
            int features = data[0].length;
            
            // Initialize centroids randomly
            centroids = new double[k][features];
            Random random = new Random();
            for (int i = 0; i < k; i++) {
                int randomIndex = random.nextInt(n);
                System.arraycopy(data[randomIndex], 0, centroids[i], 0, features);
            }
            
            int[] assignments = new int[n];
            
            for (int iter = 0; iter < maxIterations; iter++) {
                boolean changed = false;
                
                // Assign points to nearest centroids
                for (int i = 0; i < n; i++) {
                    int nearestCentroid = findNearestCentroid(data[i]);
                    if (assignments[i] != nearestCentroid) {
                        assignments[i] = nearestCentroid;
                        changed = true;
                    }
                }
                
                if (!changed) break;
                
                // Update centroids
                updateCentroids(data, assignments);
            }
            
            return assignments;
        }
        
        private int findNearestCentroid(double[] point) {
            double minDistance = Double.MAX_VALUE;
            int nearestCentroid = 0;
            
            for (int i = 0; i < k; i++) {
                double distance = euclideanDistance(point, centroids[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroid = i;
                }
            }
            
            return nearestCentroid;
        }
        
        private void updateCentroids(double[][] data, int[] assignments) {
            int features = data[0].length;
            int[] counts = new int[k];
            
            // Reset centroids
            for (int i = 0; i < k; i++) {
                Arrays.fill(centroids[i], 0);
            }
            
            // Sum points for each cluster
            for (int i = 0; i < data.length; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < features; j++) {
                    centroids[cluster][j] += data[i][j];
                }
            }
            
            // Average to get new centroids
            for (int i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    for (int j = 0; j < features; j++) {
                        centroids[i][j] /= counts[i];
                    }
                }
            }
        }
        
        public double[][] getCentroids() { return centroids; }
    }
    
    /**
     * K-Nearest Neighbors Classifier
     */
    public static class KNNClassifier {
        private double[][] trainingData;
        private int[] trainingLabels;
        private int k;
        
        public KNNClassifier(int k) {
            this.k = k;
        }
        
        /**
         * Train the KNN classifier
         * @param data Training data
         * @param labels Training labels
         */
        public void fit(double[][] data, int[] labels) {
            this.trainingData = data;
            this.trainingLabels = labels;
        }
        
        /**
         * Predict class for a single instance
         * @param instance Test instance
         * @return Predicted class
         */
        public int predict(double[] instance) {
            // Calculate distances to all training points
            List<DistanceLabel> distances = new ArrayList<>();
            
            for (int i = 0; i < trainingData.length; i++) {
                double distance = euclideanDistance(instance, trainingData[i]);
                distances.add(new DistanceLabel(distance, trainingLabels[i]));
            }
            
            // Sort by distance
            distances.sort(Comparator.comparingDouble(dl -> dl.distance));
            
            // Count votes from k nearest neighbors
            Map<Integer, Integer> votes = new HashMap<>();
            for (int i = 0; i < Math.min(k, distances.size()); i++) {
                int label = distances.get(i).label;
                votes.put(label, votes.getOrDefault(label, 0) + 1);
            }
            
            // Return most frequent class
            return votes.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .get().getKey();
        }
        
        private static class DistanceLabel {
            double distance;
            int label;
            
            DistanceLabel(double distance, int label) {
                this.distance = distance;
                this.label = label;
            }
        }
    }
    
    /**
     * Calculate Euclidean distance between two points
     * @param point1 First point
     * @param point2 Second point
     * @return Euclidean distance
     */
    public static double euclideanDistance(double[] point1, double[] point2) {
        if (point1.length != point2.length) {
            throw new IllegalArgumentException("Points must have same dimensions");
        }
        
        double sum = 0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Calculate Mean Squared Error
     * @param actual Actual values
     * @param predicted Predicted values
     * @return MSE
     */
    public static double meanSquaredError(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            double diff = actual[i] - predicted[i];
            sum += diff * diff;
        }
        
        return sum / actual.length;
    }
    
    /**
     * Calculate accuracy for classification
     * @param actual Actual labels
     * @param predicted Predicted labels
     * @return Accuracy (0 to 1)
     */
    public static double accuracy(int[] actual, int[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] == predicted[i]) {
                correct++;
            }
        }
        
        return (double) correct / actual.length;
    }
    
    public static void main(String[] args) {
        System.out.println("Machine Learning Algorithms:");
        System.out.println("============================");
        
        // Linear Regression Example
        System.out.println("Linear Regression:");
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10}; // y = 2x
        
        LinearRegression lr = new LinearRegression();
        lr.fit(x, y);
        
        System.out.println("Slope: " + lr.getSlope());
        System.out.println("Intercept: " + lr.getIntercept());
        System.out.println("Prediction for x=6: " + lr.predict(6));
        
        // K-Means Clustering Example
        System.out.println("\nK-Means Clustering:");
        double[][] clusterData = {
            {1, 2}, {1.5, 1.8}, {5, 8}, {8, 8}, {1, 0.6}, {9, 11}
        };
        
        KMeans kmeans = new KMeans(2, 100);
        int[] clusters = kmeans.fit(clusterData);
        
        System.out.println("Cluster assignments: " + Arrays.toString(clusters));
        
        // KNN Classification Example
        System.out.println("\nK-Nearest Neighbors:");
        double[][] trainData = {
            {1, 2}, {2, 3}, {3, 3}, {6, 6}, {7, 7}, {8, 6}
        };
        int[] trainLabels = {0, 0, 0, 1, 1, 1}; // Two classes: 0 and 1
        
        KNNClassifier knn = new KNNClassifier(3);
        knn.fit(trainData, trainLabels);
        
        double[] testPoint = {2, 2};
        int prediction = knn.predict(testPoint);
        System.out.println("Prediction for point [2, 2]: class " + prediction);
        
        // Calculate some metrics
        double[] actualValues = {2, 4, 6, 8, 10};
        double[] predictedValues = {2.1, 3.9, 6.1, 7.8, 10.2};
        double mse = meanSquaredError(actualValues, predictedValues);
        System.out.println("\nMean Squared Error: " + mse);
        
        int[] actualLabels = {0, 0, 1, 1, 1};
        int[] predictedLabels = {0, 1, 1, 1, 0};
        double acc = accuracy(actualLabels, predictedLabels);
        System.out.println("Classification Accuracy: " + acc);
    }
}
