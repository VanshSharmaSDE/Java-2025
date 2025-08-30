package Algorithms.MachineLearning;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * Comprehensive Machine Learning Algorithms
 * Neural networks, decision trees, clustering, and deep learning
 */
public class MachineLearningAlgorithms {
    
    /**
     * Multi-layer Perceptron Neural Network
     */
    public static class NeuralNetwork {
        
        public static class Matrix {
            private double[][] data;
            private int rows, cols;
            
            public Matrix(int rows, int cols) {
                this.rows = rows;
                this.cols = cols;
                this.data = new double[rows][cols];
            }
            
            public Matrix(double[][] data) {
                this.rows = data.length;
                this.cols = data[0].length;
                this.data = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(data[i], 0, this.data[i], 0, cols);
                }
            }
            
            public double get(int i, int j) { return data[i][j]; }
            public void set(int i, int j, double value) { data[i][j] = value; }
            public int getRows() { return rows; }
            public int getCols() { return cols; }
            
            public Matrix multiply(Matrix other) {
                if (this.cols != other.rows) {
                    throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
                }
                
                Matrix result = new Matrix(this.rows, other.cols);
                for (int i = 0; i < this.rows; i++) {
                    for (int j = 0; j < other.cols; j++) {
                        double sum = 0;
                        for (int k = 0; k < this.cols; k++) {
                            sum += this.data[i][k] * other.data[k][j];
                        }
                        result.data[i][j] = sum;
                    }
                }
                return result;
            }
            
            public Matrix add(Matrix other) {
                Matrix result = new Matrix(rows, cols);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[i][j] = this.data[i][j] + other.data[i][j];
                    }
                }
                return result;
            }
            
            public Matrix subtract(Matrix other) {
                Matrix result = new Matrix(rows, cols);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[i][j] = this.data[i][j] - other.data[i][j];
                    }
                }
                return result;
            }
            
            public Matrix transpose() {
                Matrix result = new Matrix(cols, rows);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[j][i] = this.data[i][j];
                    }
                }
                return result;
            }
            
            public Matrix hadamard(Matrix other) {
                Matrix result = new Matrix(rows, cols);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[i][j] = this.data[i][j] * other.data[i][j];
                    }
                }
                return result;
            }
            
            public void randomize(double min, double max) {
                Random random = new Random();
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        data[i][j] = min + (max - min) * random.nextDouble();
                    }
                }
            }
            
            public Matrix applyFunction(java.util.function.Function<Double, Double> func) {
                Matrix result = new Matrix(rows, cols);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[i][j] = func.apply(this.data[i][j]);
                    }
                }
                return result;
            }
        }
        
        public static class Layer {
            private Matrix weights;
            private Matrix biases;
            private Matrix lastInput;
            private Matrix lastOutput;
            
            public Layer(int inputSize, int outputSize) {
                weights = new Matrix(outputSize, inputSize);
                biases = new Matrix(outputSize, 1);
                weights.randomize(-1.0, 1.0);
                biases.randomize(-1.0, 1.0);
            }
            
            public Matrix forward(Matrix input) {
                lastInput = input;
                Matrix weightedSum = weights.multiply(input).add(biases);
                lastOutput = weightedSum.applyFunction(x -> sigmoid(x));
                return lastOutput;
            }
            
            public Matrix backward(Matrix error, double learningRate) {
                // Calculate gradient of activation function
                Matrix sigmoidDerivative = lastOutput.applyFunction(x -> x * (1 - x));
                Matrix delta = error.hadamard(sigmoidDerivative);
                
                // Update weights and biases
                Matrix weightGradient = delta.multiply(lastInput.transpose());
                Matrix biasGradient = delta;
                
                weights = weights.subtract(weightGradient.applyFunction(x -> x * learningRate));
                biases = biases.subtract(biasGradient.applyFunction(x -> x * learningRate));
                
                // Return error for previous layer
                return weights.transpose().multiply(delta);
            }
            
            private static double sigmoid(double x) {
                return 1.0 / (1.0 + Math.exp(-x));
            }
        }
        
        private List<Layer> layers;
        private double learningRate;
        
        public NeuralNetwork(int[] layerSizes, double learningRate) {
            this.learningRate = learningRate;
            this.layers = new ArrayList<>();
            
            for (int i = 0; i < layerSizes.length - 1; i++) {
                layers.add(new Layer(layerSizes[i], layerSizes[i + 1]));
            }
        }
        
        public Matrix predict(Matrix input) {
            Matrix output = input;
            for (Layer layer : layers) {
                output = layer.forward(output);
            }
            return output;
        }
        
        public void train(Matrix input, Matrix target) {
            // Forward pass
            Matrix output = predict(input);
            
            // Calculate error
            Matrix error = target.subtract(output);
            
            // Backward pass
            for (int i = layers.size() - 1; i >= 0; i--) {
                error = layers.get(i).backward(error, learningRate);
            }
        }
        
        public double calculateLoss(Matrix output, Matrix target) {
            double loss = 0;
            for (int i = 0; i < output.getRows(); i++) {
                for (int j = 0; j < output.getCols(); j++) {
                    double diff = output.get(i, j) - target.get(i, j);
                    loss += diff * diff;
                }
            }
            return loss / 2.0;
        }
    }
    
    /**
     * K-Means Clustering Algorithm
     */
    public static class KMeansClustering {
        
        public static class Point {
            public final double[] coordinates;
            public int cluster;
            
            public Point(double... coordinates) {
                this.coordinates = coordinates.clone();
                this.cluster = -1;
            }
            
            public double distanceTo(Point other) {
                double sum = 0;
                for (int i = 0; i < coordinates.length; i++) {
                    double diff = coordinates[i] - other.coordinates[i];
                    sum += diff * diff;
                }
                return Math.sqrt(sum);
            }
            
            public Point add(Point other) {
                double[] newCoords = new double[coordinates.length];
                for (int i = 0; i < coordinates.length; i++) {
                    newCoords[i] = coordinates[i] + other.coordinates[i];
                }
                return new Point(newCoords);
            }
            
            public Point divide(double divisor) {
                double[] newCoords = new double[coordinates.length];
                for (int i = 0; i < coordinates.length; i++) {
                    newCoords[i] = coordinates[i] / divisor;
                }
                return new Point(newCoords);
            }
            
            public String toString() {
                return Arrays.toString(coordinates) + " (cluster " + cluster + ")";
            }
        }
        
        public static class KMeansResult {
            public final List<Point> points;
            public final List<Point> centroids;
            public final int iterations;
            
            public KMeansResult(List<Point> points, List<Point> centroids, int iterations) {
                this.points = new ArrayList<>(points);
                this.centroids = new ArrayList<>(centroids);
                this.iterations = iterations;
            }
        }
        
        public static KMeansResult cluster(List<Point> points, int k, int maxIterations) {
            Random random = new Random();
            
            // Initialize centroids randomly
            List<Point> centroids = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                Point randomPoint = points.get(random.nextInt(points.size()));
                centroids.add(new Point(randomPoint.coordinates));
            }
            
            boolean converged = false;
            int iteration = 0;
            
            while (!converged && iteration < maxIterations) {
                // Assign points to nearest centroids
                for (Point point : points) {
                    double minDistance = Double.MAX_VALUE;
                    int nearestCluster = 0;
                    
                    for (int i = 0; i < centroids.size(); i++) {
                        double distance = point.distanceTo(centroids.get(i));
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearestCluster = i;
                        }
                    }
                    
                    point.cluster = nearestCluster;
                }
                
                // Update centroids
                List<Point> newCentroids = new ArrayList<>();
                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = points.stream()
                        .filter(p -> p.cluster == i)
                        .collect(Collectors.toList());
                    
                    if (!clusterPoints.isEmpty()) {
                        Point sum = clusterPoints.stream()
                            .reduce(new Point(new double[points.get(0).coordinates.length]), Point::add);
                        newCentroids.add(sum.divide(clusterPoints.size()));
                    } else {
                        newCentroids.add(centroids.get(i)); // Keep old centroid if no points assigned
                    }
                }
                
                // Check for convergence
                converged = true;
                for (int i = 0; i < k; i++) {
                    if (centroids.get(i).distanceTo(newCentroids.get(i)) > 1e-6) {
                        converged = false;
                        break;
                    }
                }
                
                centroids = newCentroids;
                iteration++;
            }
            
            return new KMeansResult(points, centroids, iteration);
        }
        
        public static double calculateWCSS(KMeansResult result) {
            double wcss = 0;
            for (Point point : result.points) {
                Point centroid = result.centroids.get(point.cluster);
                double distance = point.distanceTo(centroid);
                wcss += distance * distance;
            }
            return wcss;
        }
    }
    
    /**
     * Decision Tree Algorithm
     */
    public static class DecisionTree {
        
        public static class Dataset {
            public final double[][] features;
            public final int[] labels;
            public final String[] featureNames;
            
            public Dataset(double[][] features, int[] labels, String[] featureNames) {
                this.features = features;
                this.labels = labels;
                this.featureNames = featureNames;
            }
            
            public int getNumSamples() { return features.length; }
            public int getNumFeatures() { return features[0].length; }
        }
        
        public static class TreeNode {
            public int featureIndex = -1;
            public double threshold = 0;
            public int prediction = -1;
            public TreeNode left = null;
            public TreeNode right = null;
            public boolean isLeaf = false;
            
            public int predict(double[] sample) {
                if (isLeaf) {
                    return prediction;
                } else {
                    if (sample[featureIndex] <= threshold) {
                        return left.predict(sample);
                    } else {
                        return right.predict(sample);
                    }
                }
            }
        }
        
        public static class DecisionTreeClassifier {
            private TreeNode root;
            private int maxDepth;
            private int minSamplesplit;
            
            public DecisionTreeClassifier(int maxDepth, int minSamplesplit) {
                this.maxDepth = maxDepth;
                this.minSamplesplit = minSamplesplit;
            }
            
            public void fit(Dataset dataset) {
                int[] indices = IntStream.range(0, dataset.getNumSamples()).toArray();
                root = buildTree(dataset, indices, 0);
            }
            
            private TreeNode buildTree(Dataset dataset, int[] indices, int depth) {
                TreeNode node = new TreeNode();
                
                // Check stopping criteria
                if (depth >= maxDepth || indices.length < minSamplesplit || isPure(dataset, indices)) {
                    node.isLeaf = true;
                    node.prediction = getMajorityClass(dataset, indices);
                    return node;
                }
                
                // Find best split
                BestSplit bestSplit = findBestSplit(dataset, indices);
                
                if (bestSplit == null) {
                    node.isLeaf = true;
                    node.prediction = getMajorityClass(dataset, indices);
                    return node;
                }
                
                node.featureIndex = bestSplit.featureIndex;
                node.threshold = bestSplit.threshold;
                
                // Split the data
                List<Integer> leftIndices = new ArrayList<>();
                List<Integer> rightIndices = new ArrayList<>();
                
                for (int idx : indices) {
                    if (dataset.features[idx][bestSplit.featureIndex] <= bestSplit.threshold) {
                        leftIndices.add(idx);
                    } else {
                        rightIndices.add(idx);
                    }
                }
                
                // Recursively build subtrees
                node.left = buildTree(dataset, leftIndices.stream().mapToInt(i -> i).toArray(), depth + 1);
                node.right = buildTree(dataset, rightIndices.stream().mapToInt(i -> i).toArray(), depth + 1);
                
                return node;
            }
            
            private boolean isPure(Dataset dataset, int[] indices) {
                if (indices.length == 0) return true;
                
                int firstLabel = dataset.labels[indices[0]];
                for (int idx : indices) {
                    if (dataset.labels[idx] != firstLabel) {
                        return false;
                    }
                }
                return true;
            }
            
            private int getMajorityClass(Dataset dataset, int[] indices) {
                Map<Integer, Integer> counts = new HashMap<>();
                for (int idx : indices) {
                    counts.merge(dataset.labels[idx], 1, Integer::sum);
                }
                
                return counts.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(0);
            }
            
            private static class BestSplit {
                int featureIndex;
                double threshold;
                double infoGain;
                
                BestSplit(int featureIndex, double threshold, double infoGain) {
                    this.featureIndex = featureIndex;
                    this.threshold = threshold;
                    this.infoGain = infoGain;
                }
            }
            
            private BestSplit findBestSplit(Dataset dataset, int[] indices) {
                double bestInfoGain = 0;
                BestSplit bestSplit = null;
                
                for (int featureIdx = 0; featureIdx < dataset.getNumFeatures(); featureIdx++) {
                    double[] featureValues = Arrays.stream(indices)
                        .mapToDouble(i -> dataset.features[i][featureIdx])
                        .sorted()
                        .toArray();
                    
                    for (int i = 0; i < featureValues.length - 1; i++) {
                        double threshold = (featureValues[i] + featureValues[i + 1]) / 2.0;
                        double infoGain = calculateInformationGain(dataset, indices, featureIdx, threshold);
                        
                        if (infoGain > bestInfoGain) {
                            bestInfoGain = infoGain;
                            bestSplit = new BestSplit(featureIdx, threshold, infoGain);
                        }
                    }
                }
                
                return bestSplit;
            }
            
            private double calculateInformationGain(Dataset dataset, int[] indices, int featureIndex, double threshold) {
                double parentEntropy = calculateEntropy(dataset, indices);
                
                List<Integer> leftIndices = new ArrayList<>();
                List<Integer> rightIndices = new ArrayList<>();
                
                for (int idx : indices) {
                    if (dataset.features[idx][featureIndex] <= threshold) {
                        leftIndices.add(idx);
                    } else {
                        rightIndices.add(idx);
                    }
                }
                
                if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
                    return 0;
                }
                
                int totalSamples = indices.length;
                double leftWeight = (double) leftIndices.size() / totalSamples;
                double rightWeight = (double) rightIndices.size() / totalSamples;
                
                double leftEntropy = calculateEntropy(dataset, leftIndices.stream().mapToInt(i -> i).toArray());
                double rightEntropy = calculateEntropy(dataset, rightIndices.stream().mapToInt(i -> i).toArray());
                
                double weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
                
                return parentEntropy - weightedEntropy;
            }
            
            private double calculateEntropy(Dataset dataset, int[] indices) {
                Map<Integer, Integer> counts = new HashMap<>();
                for (int idx : indices) {
                    counts.merge(dataset.labels[idx], 1, Integer::sum);
                }
                
                double entropy = 0;
                int totalSamples = indices.length;
                
                for (int count : counts.values()) {
                    if (count > 0) {
                        double probability = (double) count / totalSamples;
                        entropy -= probability * Math.log(probability) / Math.log(2);
                    }
                }
                
                return entropy;
            }
            
            public int predict(double[] sample) {
                return root.predict(sample);
            }
            
            public double[] predict(double[][] samples) {
                return Arrays.stream(samples)
                    .mapToDouble(sample -> predict(sample))
                    .toArray();
            }
        }
    }
    
    /**
     * Support Vector Machine (SVM) Algorithm
     */
    public static class SupportVectorMachine {
        
        public static class SVMModel {
            private double[][] supportVectors;
            private double[] alphas;
            private double bias;
            private double[] labels;
            private double C; // Regularization parameter
            private double gamma; // RBF kernel parameter
            
            public SVMModel(double C, double gamma) {
                this.C = C;
                this.gamma = gamma;
            }
            
            public void train(double[][] X, double[] y, int maxIterations) {
                int n = X.length;
                alphas = new double[n];
                bias = 0;
                
                // Simplified SMO algorithm
                for (int iter = 0; iter < maxIterations; iter++) {
                    int numChanged = 0;
                    
                    for (int i = 0; i < n; i++) {
                        double Ei = predict(X[i]) - y[i];
                        
                        if ((y[i] * Ei < -1e-3 && alphas[i] < C) || (y[i] * Ei > 1e-3 && alphas[i] > 0)) {
                            int j = selectSecondAlpha(i, n);
                            double Ej = predict(X[j]) - y[j];
                            
                            double oldAlphaI = alphas[i];
                            double oldAlphaJ = alphas[j];
                            
                            // Compute bounds
                            double L, H;
                            if (y[i] != y[j]) {
                                L = Math.max(0, alphas[j] - alphas[i]);
                                H = Math.min(C, C + alphas[j] - alphas[i]);
                            } else {
                                L = Math.max(0, alphas[i] + alphas[j] - C);
                                H = Math.min(C, alphas[i] + alphas[j]);
                            }
                            
                            if (L == H) continue;
                            
                            // Compute eta
                            double eta = 2 * rbfKernel(X[i], X[j]) - rbfKernel(X[i], X[i]) - rbfKernel(X[j], X[j]);
                            if (eta >= 0) continue;
                            
                            // Update alphas
                            alphas[j] = oldAlphaJ - (y[j] * (Ei - Ej)) / eta;
                            alphas[j] = Math.max(L, Math.min(H, alphas[j]));
                            
                            if (Math.abs(alphas[j] - oldAlphaJ) < 1e-5) continue;
                            
                            alphas[i] = oldAlphaI + y[i] * y[j] * (oldAlphaJ - alphas[j]);
                            
                            // Update bias
                            double b1 = bias - Ei - y[i] * (alphas[i] - oldAlphaI) * rbfKernel(X[i], X[i])
                                       - y[j] * (alphas[j] - oldAlphaJ) * rbfKernel(X[i], X[j]);
                            double b2 = bias - Ej - y[i] * (alphas[i] - oldAlphaI) * rbfKernel(X[i], X[j])
                                       - y[j] * (alphas[j] - oldAlphaJ) * rbfKernel(X[j], X[j]);
                            
                            if (0 < alphas[i] && alphas[i] < C) {
                                bias = b1;
                            } else if (0 < alphas[j] && alphas[j] < C) {
                                bias = b2;
                            } else {
                                bias = (b1 + b2) / 2;
                            }
                            
                            numChanged++;
                        }
                    }
                    
                    if (numChanged == 0) break;
                }
                
                // Extract support vectors
                List<double[]> svList = new ArrayList<>();
                List<Double> alphaList = new ArrayList<>();
                List<Double> labelList = new ArrayList<>();
                
                for (int i = 0; i < n; i++) {
                    if (alphas[i] > 1e-5) {
                        svList.add(X[i]);
                        alphaList.add(alphas[i]);
                        labelList.add(y[i]);
                    }
                }
                
                supportVectors = svList.toArray(new double[0][]);
                alphas = alphaList.stream().mapToDouble(Double::doubleValue).toArray();
                labels = labelList.stream().mapToDouble(Double::doubleValue).toArray();
            }
            
            private int selectSecondAlpha(int i, int n) {
                Random random = new Random();
                int j;
                do {
                    j = random.nextInt(n);
                } while (j == i);
                return j;
            }
            
            private double rbfKernel(double[] x1, double[] x2) {
                double sum = 0;
                for (int i = 0; i < x1.length; i++) {
                    sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
                }
                return Math.exp(-gamma * sum);
            }
            
            public double predict(double[] x) {
                double result = 0;
                for (int i = 0; i < supportVectors.length; i++) {
                    result += alphas[i] * labels[i] * rbfKernel(x, supportVectors[i]);
                }
                return result + bias;
            }
            
            public double[] predict(double[][] X) {
                return Arrays.stream(X).mapToDouble(this::predict).toArray();
            }
        }
    }
    
    /**
     * Random Forest Algorithm
     */
    public static class RandomForest {
        
        public static class RandomForestClassifier {
            private List<DecisionTree.DecisionTreeClassifier> trees;
            private int numTrees;
            private int maxFeatures;
            private Random random;
            
            public RandomForestClassifier(int numTrees, int maxDepth, int maxFeatures) {
                this.numTrees = numTrees;
                this.maxFeatures = maxFeatures;
                this.trees = new ArrayList<>();
                this.random = new Random();
                
                for (int i = 0; i < numTrees; i++) {
                    trees.add(new DecisionTree.DecisionTreeClassifier(maxDepth, 2));
                }
            }
            
            public void fit(DecisionTree.Dataset dataset) {
                for (DecisionTree.DecisionTreeClassifier tree : trees) {
                    // Bootstrap sampling
                    DecisionTree.Dataset bootstrapSample = createBootstrapSample(dataset);
                    
                    // Train tree on bootstrap sample
                    tree.fit(bootstrapSample);
                }
            }
            
            private DecisionTree.Dataset createBootstrapSample(DecisionTree.Dataset dataset) {
                int n = dataset.getNumSamples();
                int[] indices = new int[n];
                
                for (int i = 0; i < n; i++) {
                    indices[i] = random.nextInt(n);
                }
                
                double[][] features = new double[n][];
                int[] labels = new int[n];
                
                for (int i = 0; i < n; i++) {
                    features[i] = dataset.features[indices[i]].clone();
                    labels[i] = dataset.labels[indices[i]];
                }
                
                return new DecisionTree.Dataset(features, labels, dataset.featureNames);
            }
            
            public int predict(double[] sample) {
                Map<Integer, Integer> votes = new HashMap<>();
                
                for (DecisionTree.DecisionTreeClassifier tree : trees) {
                    int prediction = tree.predict(sample);
                    votes.merge(prediction, 1, Integer::sum);
                }
                
                return votes.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(0);
            }
            
            public double[] predict(double[][] samples) {
                return Arrays.stream(samples)
                    .mapToDouble(sample -> predict(sample))
                    .toArray();
            }
            
            public double calculateAccuracy(double[][] testFeatures, int[] testLabels) {
                int correct = 0;
                for (int i = 0; i < testFeatures.length; i++) {
                    if (predict(testFeatures[i]) == testLabels[i]) {
                        correct++;
                    }
                }
                return (double) correct / testFeatures.length;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Machine Learning Algorithms Demo:");
        System.out.println("=================================");
        
        // Neural Network demonstration
        System.out.println("1. Neural Network (XOR Problem):");
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 4, 1}, 0.5);
        
        // XOR training data
        NeuralNetwork.Matrix[] inputs = {
            new NeuralNetwork.Matrix(new double[][]{{0}, {0}}),
            new NeuralNetwork.Matrix(new double[][]{{0}, {1}}),
            new NeuralNetwork.Matrix(new double[][]{{1}, {0}}),
            new NeuralNetwork.Matrix(new double[][]{{1}, {1}})
        };
        
        NeuralNetwork.Matrix[] targets = {
            new NeuralNetwork.Matrix(new double[][]{{0}}),
            new NeuralNetwork.Matrix(new double[][]{{1}}),
            new NeuralNetwork.Matrix(new double[][]{{1}}),
            new NeuralNetwork.Matrix(new double[][]{{0}})
        };
        
        // Train the network
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.train(inputs[i], targets[i]);
            }
        }
        
        // Test the network
        System.out.println("XOR Results:");
        for (int i = 0; i < inputs.length; i++) {
            NeuralNetwork.Matrix output = nn.predict(inputs[i]);
            System.out.printf("Input: [%.0f, %.0f] -> Output: %.3f (Target: %.0f)\n",
                            inputs[i].get(0, 0), inputs[i].get(1, 0), 
                            output.get(0, 0), targets[i].get(0, 0));
        }
        
        // K-Means Clustering demonstration
        System.out.println("\n2. K-Means Clustering:");
        List<KMeansClustering.Point> points = Arrays.asList(
            new KMeansClustering.Point(1.0, 2.0),
            new KMeansClustering.Point(1.5, 1.8),
            new KMeansClustering.Point(5.0, 8.0),
            new KMeansClustering.Point(8.0, 8.0),
            new KMeansClustering.Point(1.0, 0.6),
            new KMeansClustering.Point(9.0, 11.0),
            new KMeansClustering.Point(8.0, 2.0),
            new KMeansClustering.Point(10.0, 2.0),
            new KMeansClustering.Point(9.0, 3.0)
        );
        
        KMeansClustering.KMeansResult result = KMeansClustering.cluster(points, 3, 100);
        System.out.println("Clustering completed in " + result.iterations + " iterations");
        System.out.println("WCSS: " + KMeansClustering.calculateWCSS(result));
        
        for (int i = 0; i < result.centroids.size(); i++) {
            System.out.println("Centroid " + i + ": " + Arrays.toString(result.centroids.get(i).coordinates));
        }
        
        // Decision Tree demonstration
        System.out.println("\n3. Decision Tree Classification:");
        double[][] features = {
            {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
            {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
            {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1}
        };
        
        int[] labels = {0, 0, 0, 1, 1, 1, 2, 2, 2}; // Three classes
        String[] featureNames = {"sepal_length", "sepal_width", "petal_length", "petal_width"};
        
        DecisionTree.Dataset dataset = new DecisionTree.Dataset(features, labels, featureNames);
        DecisionTree.DecisionTreeClassifier dt = new DecisionTree.DecisionTreeClassifier(5, 2);
        dt.fit(dataset);
        
        System.out.println("Decision tree trained on iris-like data");
        double[] predictions = dt.predict(features);
        System.out.println("Training accuracy: " + 
            Arrays.stream(predictions).mapToInt(x -> (int)x)
                  .zip(Arrays.stream(labels), (p, l) -> p == l ? 1 : 0)
                  .sum() / (double) labels.length);
        
        // Random Forest demonstration
        System.out.println("\n4. Random Forest Classification:");
        RandomForest.RandomForestClassifier rf = new RandomForest.RandomForestClassifier(10, 5, 2);
        rf.fit(dataset);
        
        double accuracy = rf.calculateAccuracy(features, labels);
        System.out.println("Random Forest training accuracy: " + accuracy);
        
        System.out.println("\nMachine learning demonstration completed!");
    }
}
