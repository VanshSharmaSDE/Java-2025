package Algorithms.Mathematics.Statistics;

import java.util.*;

/**
 * Statistical Algorithms and Formulas
 */
public class StatisticalAlgorithms {
    
    /**
     * Calculate mean (average) of a dataset
     * @param data Array of values
     * @return Mean value
     */
    public static double calculateMean(double[] data) {
        double sum = 0;
        for (double value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    /**
     * Calculate median of a dataset
     * @param data Array of values
     * @return Median value
     */
    public static double calculateMedian(double[] data) {
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    /**
     * Calculate mode (most frequent value) of a dataset
     * @param data Array of values
     * @return Mode value
     */
    public static double calculateMode(double[] data) {
        Map<Double, Integer> frequency = new HashMap<>();
        
        for (double value : data) {
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }
        
        double mode = data[0];
        int maxCount = 0;
        
        for (Map.Entry<Double, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mode = entry.getKey();
            }
        }
        
        return mode;
    }
    
    /**
     * Calculate variance of a dataset
     * @param data Array of values
     * @param population True for population variance, false for sample variance
     * @return Variance
     */
    public static double calculateVariance(double[] data, boolean population) {
        double mean = calculateMean(data);
        double sumSquaredDiff = 0;
        
        for (double value : data) {
            sumSquaredDiff += Math.pow(value - mean, 2);
        }
        
        int denominator = population ? data.length : data.length - 1;
        return sumSquaredDiff / denominator;
    }
    
    /**
     * Calculate standard deviation
     * @param data Array of values
     * @param population True for population std dev, false for sample std dev
     * @return Standard deviation
     */
    public static double calculateStandardDeviation(double[] data, boolean population) {
        return Math.sqrt(calculateVariance(data, population));
    }
    
    /**
     * Calculate correlation coefficient between two datasets
     * @param x First dataset
     * @param y Second dataset
     * @return Pearson correlation coefficient
     */
    public static double calculateCorrelation(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double meanX = calculateMean(x);
        double meanY = calculateMean(y);
        
        double numerator = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;
        
        for (int i = 0; i < x.length; i++) {
            double xDiff = x[i] - meanX;
            double yDiff = y[i] - meanY;
            
            numerator += xDiff * yDiff;
            sumXSquared += xDiff * xDiff;
            sumYSquared += yDiff * yDiff;
        }
        
        double denominator = Math.sqrt(sumXSquared * sumYSquared);
        return denominator == 0 ? 0 : numerator / denominator;
    }
    
    /**
     * Calculate Z-score for a value
     * @param value The value
     * @param mean Mean of the distribution
     * @param standardDeviation Standard deviation of the distribution
     * @return Z-score
     */
    public static double calculateZScore(double value, double mean, double standardDeviation) {
        return (value - mean) / standardDeviation;
    }
    
    /**
     * Calculate percentile of a dataset
     * @param data Array of values
     * @param percentile Percentile to calculate (0-100)
     * @return Value at the given percentile
     */
    public static double calculatePercentile(double[] data, double percentile) {
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be between 0 and 100");
        }
        
        double index = (percentile / 100.0) * (sorted.length - 1);
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);
        
        if (lowerIndex == upperIndex) {
            return sorted[lowerIndex];
        } else {
            double weight = index - lowerIndex;
            return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
        }
    }
    
    /**
     * Perform linear regression and return slope and intercept
     * @param x Independent variable
     * @param y Dependent variable
     * @return Array containing [slope, intercept, r-squared]
     */
    public static double[] linearRegression(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double meanX = calculateMean(x);
        double meanY = calculateMean(y);
        
        double numerator = 0;
        double denominator = 0;
        
        for (int i = 0; i < x.length; i++) {
            double xDiff = x[i] - meanX;
            numerator += xDiff * (y[i] - meanY);
            denominator += xDiff * xDiff;
        }
        
        double slope = numerator / denominator;
        double intercept = meanY - slope * meanX;
        
        // Calculate R-squared
        double totalSumSquares = 0;
        double residualSumSquares = 0;
        
        for (int i = 0; i < x.length; i++) {
            double predicted = slope * x[i] + intercept;
            totalSumSquares += Math.pow(y[i] - meanY, 2);
            residualSumSquares += Math.pow(y[i] - predicted, 2);
        }
        
        double rSquared = 1 - (residualSumSquares / totalSumSquares);
        
        return new double[]{slope, intercept, rSquared};
    }
    
    /**
     * Normal distribution probability density function
     * @param x Value
     * @param mean Mean of distribution
     * @param standardDeviation Standard deviation of distribution
     * @return Probability density
     */
    public static double normalPDF(double x, double mean, double standardDeviation) {
        double coefficient = 1.0 / (standardDeviation * Math.sqrt(2 * Math.PI));
        double exponent = -0.5 * Math.pow((x - mean) / standardDeviation, 2);
        return coefficient * Math.exp(exponent);
    }
    
    /**
     * Approximate normal distribution cumulative distribution function
     * @param x Value
     * @param mean Mean of distribution
     * @param standardDeviation Standard deviation of distribution
     * @return Cumulative probability
     */
    public static double normalCDF(double x, double mean, double standardDeviation) {
        double z = (x - mean) / standardDeviation;
        return 0.5 * (1 + erf(z / Math.sqrt(2)));
    }
    
    /**
     * Approximate error function using series expansion
     * @param x Input value
     * @return Error function value
     */
    private static double erf(double x) {
        // Approximation using series expansion
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;
        
        int sign = x < 0 ? -1 : 1;
        x = Math.abs(x);
        
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        
        return sign * y;
    }
    
    /**
     * Calculate confidence interval for mean
     * @param data Sample data
     * @param confidenceLevel Confidence level (e.g., 0.95 for 95%)
     * @return Array containing [lower bound, upper bound]
     */
    public static double[] confidenceInterval(double[] data, double confidenceLevel) {
        double mean = calculateMean(data);
        double stdDev = calculateStandardDeviation(data, false);
        double n = data.length;
        
        // Using t-distribution approximation (for large n, approaches normal)
        double alpha = 1 - confidenceLevel;
        double tValue = 1.96; // Approximation for 95% confidence (should use t-table for exact values)
        
        if (confidenceLevel == 0.90) tValue = 1.645;
        else if (confidenceLevel == 0.95) tValue = 1.96;
        else if (confidenceLevel == 0.99) tValue = 2.576;
        
        double marginOfError = tValue * (stdDev / Math.sqrt(n));
        
        return new double[]{mean - marginOfError, mean + marginOfError};
    }
    
    public static void main(String[] args) {
        System.out.println("Statistical Algorithms:");
        System.out.println("======================");
        
        // Sample dataset
        double[] data = {2, 4, 4, 4, 5, 5, 7, 9};
        
        System.out.println("Dataset: " + Arrays.toString(data));
        
        // Basic statistics
        System.out.println("\nDescriptive Statistics:");
        System.out.println("Mean: " + calculateMean(data));
        System.out.println("Median: " + calculateMedian(data));
        System.out.println("Mode: " + calculateMode(data));
        System.out.println("Variance (sample): " + calculateVariance(data, false));
        System.out.println("Standard Deviation (sample): " + calculateStandardDeviation(data, false));
        
        // Percentiles
        System.out.println("\nPercentiles:");
        System.out.println("25th percentile: " + calculatePercentile(data, 25));
        System.out.println("50th percentile (median): " + calculatePercentile(data, 50));
        System.out.println("75th percentile: " + calculatePercentile(data, 75));
        
        // Correlation and regression
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};
        
        System.out.println("\nCorrelation and Regression:");
        System.out.println("X: " + Arrays.toString(x));
        System.out.println("Y: " + Arrays.toString(y));
        System.out.println("Correlation: " + calculateCorrelation(x, y));
        
        double[] regression = linearRegression(x, y);
        System.out.println("Linear Regression:");
        System.out.println("Slope: " + regression[0]);
        System.out.println("Intercept: " + regression[1]);
        System.out.println("R-squared: " + regression[2]);
        
        // Normal distribution
        System.out.println("\nNormal Distribution (μ=0, σ=1):");
        double value = 1.0;
        System.out.println("PDF at x=" + value + ": " + normalPDF(value, 0, 1));
        System.out.println("CDF at x=" + value + ": " + normalCDF(value, 0, 1));
        
        // Z-score
        double sampleValue = 6;
        double zScore = calculateZScore(sampleValue, calculateMean(data), calculateStandardDeviation(data, false));
        System.out.println("\nZ-score for value " + sampleValue + ": " + zScore);
        
        // Confidence interval
        double[] ci = confidenceInterval(data, 0.95);
        System.out.println("\n95% Confidence Interval for mean: [" + ci[0] + ", " + ci[1] + "]");
    }
}
