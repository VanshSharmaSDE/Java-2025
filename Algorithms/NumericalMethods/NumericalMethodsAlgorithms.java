package Algorithms.NumericalMethods;

import java.util.*;
import java.util.function.*;

/**
 * Comprehensive Numerical Methods and Scientific Computing Algorithms
 * Root finding, integration, differential equations, linear algebra
 */
public class NumericalMethodsAlgorithms {
    
    /**
     * Root Finding Algorithms
     */
    public static class RootFinding {
        
        public static class RootResult {
            public final double root;
            public final double error;
            public final int iterations;
            public final boolean converged;
            
            public RootResult(double root, double error, int iterations, boolean converged) {
                this.root = root;
                this.error = error;
                this.iterations = iterations;
                this.converged = converged;
            }
            
            public String toString() {
                return String.format("Root[value=%.6f, error=%.2e, iterations=%d, converged=%s]", 
                                   root, error, iterations, converged);
            }
        }
        
        public static RootResult bisectionMethod(Function<Double, Double> f, double a, double b, double tolerance, int maxIterations) {
            if (f.apply(a) * f.apply(b) > 0) {
                throw new IllegalArgumentException("Function must have opposite signs at endpoints");
            }
            
            double fa = f.apply(a);
            double fb = f.apply(b);
            
            for (int i = 0; i < maxIterations; i++) {
                double c = (a + b) / 2.0;
                double fc = f.apply(c);
                
                if (Math.abs(fc) < tolerance || (b - a) / 2.0 < tolerance) {
                    return new RootResult(c, Math.abs(fc), i + 1, true);
                }
                
                if (fa * fc < 0) {
                    b = c;
                    fb = fc;
                } else {
                    a = c;
                    fa = fc;
                }
            }
            
            return new RootResult((a + b) / 2.0, Math.abs(f.apply((a + b) / 2.0)), maxIterations, false);
        }
        
        public static RootResult newtonRaphsonMethod(Function<Double, Double> f, Function<Double, Double> df, 
                                                   double x0, double tolerance, int maxIterations) {
            double x = x0;
            
            for (int i = 0; i < maxIterations; i++) {
                double fx = f.apply(x);
                double dfx = df.apply(x);
                
                if (Math.abs(dfx) < 1e-12) {
                    throw new RuntimeException("Derivative too small, method may not converge");
                }
                
                double xNew = x - fx / dfx;
                double error = Math.abs(xNew - x);
                
                if (error < tolerance) {
                    return new RootResult(xNew, Math.abs(f.apply(xNew)), i + 1, true);
                }
                
                x = xNew;
            }
            
            return new RootResult(x, Math.abs(f.apply(x)), maxIterations, false);
        }
        
        public static RootResult secantMethod(Function<Double, Double> f, double x0, double x1, 
                                            double tolerance, int maxIterations) {
            double xPrev = x0;
            double x = x1;
            
            for (int i = 0; i < maxIterations; i++) {
                double fx = f.apply(x);
                double fxPrev = f.apply(xPrev);
                
                if (Math.abs(fx - fxPrev) < 1e-12) {
                    throw new RuntimeException("Function values too close, method may not converge");
                }
                
                double xNew = x - fx * (x - xPrev) / (fx - fxPrev);
                double error = Math.abs(xNew - x);
                
                if (error < tolerance) {
                    return new RootResult(xNew, Math.abs(f.apply(xNew)), i + 1, true);
                }
                
                xPrev = x;
                x = xNew;
            }
            
            return new RootResult(x, Math.abs(f.apply(x)), maxIterations, false);
        }
        
        public static RootResult falsePositionMethod(Function<Double, Double> f, double a, double b, 
                                                   double tolerance, int maxIterations) {
            if (f.apply(a) * f.apply(b) > 0) {
                throw new IllegalArgumentException("Function must have opposite signs at endpoints");
            }
            
            for (int i = 0; i < maxIterations; i++) {
                double fa = f.apply(a);
                double fb = f.apply(b);
                
                double c = (a * fb - b * fa) / (fb - fa);
                double fc = f.apply(c);
                
                if (Math.abs(fc) < tolerance) {
                    return new RootResult(c, Math.abs(fc), i + 1, true);
                }
                
                if (fa * fc < 0) {
                    b = c;
                } else {
                    a = c;
                }
            }
            
            double fa = f.apply(a);
            double fb = f.apply(b);
            double c = (a * fb - b * fa) / (fb - fa);
            
            return new RootResult(c, Math.abs(f.apply(c)), maxIterations, false);
        }
    }
    
    /**
     * Numerical Integration (Quadrature) Methods
     */
    public static class NumericalIntegration {
        
        public static class IntegrationResult {
            public final double integral;
            public final double error;
            public final int evaluations;
            
            public IntegrationResult(double integral, double error, int evaluations) {
                this.integral = integral;
                this.error = error;
                this.evaluations = evaluations;
            }
            
            public String toString() {
                return String.format("Integral[value=%.6f, error=%.2e, evaluations=%d]", 
                                   integral, error, evaluations);
            }
        }
        
        public static IntegrationResult trapezoidalRule(Function<Double, Double> f, double a, double b, int n) {
            double h = (b - a) / n;
            double sum = (f.apply(a) + f.apply(b)) / 2.0;
            
            for (int i = 1; i < n; i++) {
                sum += f.apply(a + i * h);
            }
            
            double integral = h * sum;
            
            // Error estimation using Richardson extrapolation
            double integral2n = adaptiveTrapezoidal(f, a, b, 2 * n);
            double error = Math.abs(integral2n - integral) / 3.0;
            
            return new IntegrationResult(integral, error, n + 1);
        }
        
        private static double adaptiveTrapezoidal(Function<Double, Double> f, double a, double b, int n) {
            double h = (b - a) / n;
            double sum = (f.apply(a) + f.apply(b)) / 2.0;
            
            for (int i = 1; i < n; i++) {
                sum += f.apply(a + i * h);
            }
            
            return h * sum;
        }
        
        public static IntegrationResult simpsonsRule(Function<Double, Double> f, double a, double b, int n) {
            if (n % 2 != 0) {
                n++; // Ensure n is even
            }
            
            double h = (b - a) / n;
            double sum = f.apply(a) + f.apply(b);
            
            // Add even terms
            for (int i = 2; i < n; i += 2) {
                sum += 2 * f.apply(a + i * h);
            }
            
            // Add odd terms
            for (int i = 1; i < n; i += 2) {
                sum += 4 * f.apply(a + i * h);
            }
            
            double integral = (h / 3.0) * sum;
            
            // Error estimation
            double integral2n = adaptiveSimpsons(f, a, b, 2 * n);
            double error = Math.abs(integral2n - integral) / 15.0;
            
            return new IntegrationResult(integral, error, n + 1);
        }
        
        private static double adaptiveSimpsons(Function<Double, Double> f, double a, double b, int n) {
            if (n % 2 != 0) n++;
            
            double h = (b - a) / n;
            double sum = f.apply(a) + f.apply(b);
            
            for (int i = 2; i < n; i += 2) {
                sum += 2 * f.apply(a + i * h);
            }
            
            for (int i = 1; i < n; i += 2) {
                sum += 4 * f.apply(a + i * h);
            }
            
            return (h / 3.0) * sum;
        }
        
        public static IntegrationResult gaussianQuadrature(Function<Double, Double> f, double a, double b, int n) {
            // Gauss-Legendre nodes and weights for common orders
            double[][] nodes = {
                {},
                {0.0},
                {-0.5773502692, 0.5773502692},
                {-0.7745966692, 0.0, 0.7745966692},
                {-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116},
                {-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459}
            };
            
            double[][] weights = {
                {},
                {2.0},
                {1.0, 1.0},
                {0.5555555556, 0.8888888889, 0.5555555556},
                {0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451},
                {0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851}
            };
            
            if (n > 5) n = 5; // Limit to available data
            if (n < 1) n = 1;
            
            double sum = 0.0;
            for (int i = 0; i < nodes[n].length; i++) {
                double x = (b - a) / 2.0 * nodes[n][i] + (a + b) / 2.0;
                sum += weights[n][i] * f.apply(x);
            }
            
            double integral = (b - a) / 2.0 * sum;
            
            // Rough error estimation
            double error = Math.abs(integral) * 1e-6 * Math.pow(2, -n);
            
            return new IntegrationResult(integral, error, nodes[n].length);
        }
        
        public static IntegrationResult monteCarloIntegration(Function<Double, Double> f, double a, double b, int n) {
            Random random = new Random();
            double sum = 0.0;
            double sumSquared = 0.0;
            
            for (int i = 0; i < n; i++) {
                double x = a + random.nextDouble() * (b - a);
                double fx = f.apply(x);
                sum += fx;
                sumSquared += fx * fx;
            }
            
            double mean = sum / n;
            double variance = (sumSquared / n - mean * mean);
            double integral = (b - a) * mean;
            double error = (b - a) * Math.sqrt(variance / n);
            
            return new IntegrationResult(integral, error, n);
        }
    }
    
    /**
     * Ordinary Differential Equation Solvers
     */
    public static class ODESolvers {
        
        public static class ODEResult {
            public final double[] x;
            public final double[] y;
            public final int steps;
            
            public ODEResult(double[] x, double[] y, int steps) {
                this.x = x.clone();
                this.y = y.clone();
                this.steps = steps;
            }
            
            public String toString() {
                return String.format("ODEResult[steps=%d, final_x=%.3f, final_y=%.6f]", 
                                   steps, x[x.length-1], y[y.length-1]);
            }
        }
        
        public static ODEResult eulerMethod(BiFunction<Double, Double, Double> f, 
                                          double x0, double y0, double xEnd, double h) {
            int n = (int) Math.ceil((xEnd - x0) / h) + 1;
            double[] x = new double[n];
            double[] y = new double[n];
            
            x[0] = x0;
            y[0] = y0;
            
            for (int i = 1; i < n; i++) {
                x[i] = x[i-1] + h;
                y[i] = y[i-1] + h * f.apply(x[i-1], y[i-1]);
            }
            
            return new ODEResult(x, y, n - 1);
        }
        
        public static ODEResult heunMethod(BiFunction<Double, Double, Double> f, 
                                         double x0, double y0, double xEnd, double h) {
            int n = (int) Math.ceil((xEnd - x0) / h) + 1;
            double[] x = new double[n];
            double[] y = new double[n];
            
            x[0] = x0;
            y[0] = y0;
            
            for (int i = 1; i < n; i++) {
                x[i] = x[i-1] + h;
                
                // Predictor step
                double yPredictor = y[i-1] + h * f.apply(x[i-1], y[i-1]);
                
                // Corrector step
                y[i] = y[i-1] + (h / 2.0) * (f.apply(x[i-1], y[i-1]) + f.apply(x[i], yPredictor));
            }
            
            return new ODEResult(x, y, n - 1);
        }
        
        public static ODEResult rungeKutta4(BiFunction<Double, Double, Double> f, 
                                          double x0, double y0, double xEnd, double h) {
            int n = (int) Math.ceil((xEnd - x0) / h) + 1;
            double[] x = new double[n];
            double[] y = new double[n];
            
            x[0] = x0;
            y[0] = y0;
            
            for (int i = 1; i < n; i++) {
                x[i] = x[i-1] + h;
                
                double k1 = h * f.apply(x[i-1], y[i-1]);
                double k2 = h * f.apply(x[i-1] + h/2, y[i-1] + k1/2);
                double k3 = h * f.apply(x[i-1] + h/2, y[i-1] + k2/2);
                double k4 = h * f.apply(x[i-1] + h, y[i-1] + k3);
                
                y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6.0;
            }
            
            return new ODEResult(x, y, n - 1);
        }
        
        public static ODEResult adamsBashforth4(BiFunction<Double, Double, Double> f, 
                                              double x0, double y0, double xEnd, double h) {
            // Use RK4 for the first few steps
            ODEResult rkStart = rungeKutta4(f, x0, y0, x0 + 3*h, h);
            
            int totalSteps = (int) Math.ceil((xEnd - x0) / h) + 1;
            double[] x = new double[totalSteps];
            double[] y = new double[totalSteps];
            
            // Copy initial values from RK4
            for (int i = 0; i < 4 && i < totalSteps; i++) {
                x[i] = rkStart.x[i];
                y[i] = rkStart.y[i];
            }
            
            // Continue with Adams-Bashforth
            for (int i = 4; i < totalSteps; i++) {
                x[i] = x[i-1] + h;
                
                double f0 = f.apply(x[i-1], y[i-1]);
                double f1 = f.apply(x[i-2], y[i-2]);
                double f2 = f.apply(x[i-3], y[i-3]);
                double f3 = f.apply(x[i-4], y[i-4]);
                
                y[i] = y[i-1] + (h / 24.0) * (55*f0 - 59*f1 + 37*f2 - 9*f3);
            }
            
            return new ODEResult(x, y, totalSteps - 1);
        }
        
        // System of ODEs solver using RK4
        public static class SystemODEResult {
            public final double[] x;
            public final double[][] y; // y[variable][time_step]
            public final int steps;
            
            public SystemODEResult(double[] x, double[][] y, int steps) {
                this.x = x.clone();
                this.y = new double[y.length][];
                for (int i = 0; i < y.length; i++) {
                    this.y[i] = y[i].clone();
                }
                this.steps = steps;
            }
        }
        
        public static SystemODEResult rungeKutta4System(Function<double[], double[]> f, 
                                                       double x0, double[] y0, double xEnd, double h) {
            int n = (int) Math.ceil((xEnd - x0) / h) + 1;
            int numVars = y0.length;
            
            double[] x = new double[n];
            double[][] y = new double[numVars][n];
            
            x[0] = x0;
            for (int j = 0; j < numVars; j++) {
                y[j][0] = y0[j];
            }
            
            for (int i = 1; i < n; i++) {
                x[i] = x[i-1] + h;
                
                // Current state
                double[] currentY = new double[numVars];
                for (int j = 0; j < numVars; j++) {
                    currentY[j] = y[j][i-1];
                }
                
                // k1
                double[] k1 = f.apply(currentY);
                
                // k2
                double[] y_k2 = new double[numVars];
                for (int j = 0; j < numVars; j++) {
                    y_k2[j] = currentY[j] + h * k1[j] / 2;
                }
                double[] k2 = f.apply(y_k2);
                
                // k3
                double[] y_k3 = new double[numVars];
                for (int j = 0; j < numVars; j++) {
                    y_k3[j] = currentY[j] + h * k2[j] / 2;
                }
                double[] k3 = f.apply(y_k3);
                
                // k4
                double[] y_k4 = new double[numVars];
                for (int j = 0; j < numVars; j++) {
                    y_k4[j] = currentY[j] + h * k3[j];
                }
                double[] k4 = f.apply(y_k4);
                
                // Update
                for (int j = 0; j < numVars; j++) {
                    y[j][i] = currentY[j] + (h / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
                }
            }
            
            return new SystemODEResult(x, y, n - 1);
        }
    }
    
    /**
     * Linear Algebra Numerical Methods
     */
    public static class LinearAlgebra {
        
        public static class Matrix {
            private final double[][] data;
            private final int rows, cols;
            
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
                    throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication");
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
            
            public double[] multiplyVector(double[] vector) {
                if (cols != vector.length) {
                    throw new IllegalArgumentException("Matrix and vector dimensions incompatible");
                }
                
                double[] result = new double[rows];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result[i] += data[i][j] * vector[j];
                    }
                }
                return result;
            }
            
            public Matrix transpose() {
                Matrix result = new Matrix(cols, rows);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result.data[j][i] = data[i][j];
                    }
                }
                return result;
            }
            
            public double determinant() {
                if (rows != cols) {
                    throw new IllegalArgumentException("Determinant only defined for square matrices");
                }
                
                return determinantRecursive(data, rows);
            }
            
            private static double determinantRecursive(double[][] matrix, int n) {
                if (n == 1) return matrix[0][0];
                if (n == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
                
                double det = 0;
                for (int col = 0; col < n; col++) {
                    double[][] minor = getMinor(matrix, 0, col, n);
                    det += Math.pow(-1, col) * matrix[0][col] * determinantRecursive(minor, n - 1);
                }
                
                return det;
            }
            
            private static double[][] getMinor(double[][] matrix, int excludeRow, int excludeCol, int n) {
                double[][] minor = new double[n-1][n-1];
                int minorRow = 0;
                
                for (int row = 0; row < n; row++) {
                    if (row == excludeRow) continue;
                    int minorCol = 0;
                    
                    for (int col = 0; col < n; col++) {
                        if (col == excludeCol) continue;
                        minor[minorRow][minorCol] = matrix[row][col];
                        minorCol++;
                    }
                    minorRow++;
                }
                
                return minor;
            }
            
            public void fillRandom() {
                Random random = new Random();
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        data[i][j] = random.nextGaussian() * 10;
                    }
                }
            }
            
            public void fillIdentity() {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        data[i][j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }
            
            public String toString() {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < Math.min(3, rows); i++) {
                    sb.append("[");
                    for (int j = 0; j < Math.min(3, cols); j++) {
                        sb.append(String.format("%8.3f", data[i][j]));
                        if (j < Math.min(3, cols) - 1) sb.append(", ");
                    }
                    if (cols > 3) sb.append(", ...");
                    sb.append("]\n");
                }
                if (rows > 3) sb.append("...\n");
                return sb.toString();
            }
        }
        
        public static double[] gaussianElimination(Matrix A, double[] b) {
            int n = A.getRows();
            Matrix augmented = new Matrix(n, n + 1);
            
            // Create augmented matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    augmented.set(i, j, A.get(i, j));
                }
                augmented.set(i, n, b[i]);
            }
            
            // Forward elimination
            for (int k = 0; k < n; k++) {
                // Find pivot
                int maxRow = k;
                for (int i = k + 1; i < n; i++) {
                    if (Math.abs(augmented.get(i, k)) > Math.abs(augmented.get(maxRow, k))) {
                        maxRow = i;
                    }
                }
                
                // Swap rows
                if (maxRow != k) {
                    for (int j = 0; j <= n; j++) {
                        double temp = augmented.get(k, j);
                        augmented.set(k, j, augmented.get(maxRow, j));
                        augmented.set(maxRow, j, temp);
                    }
                }
                
                // Eliminate column
                for (int i = k + 1; i < n; i++) {
                    double factor = augmented.get(i, k) / augmented.get(k, k);
                    for (int j = k; j <= n; j++) {
                        augmented.set(i, j, augmented.get(i, j) - factor * augmented.get(k, j));
                    }
                }
            }
            
            // Back substitution
            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--) {
                x[i] = augmented.get(i, n);
                for (int j = i + 1; j < n; j++) {
                    x[i] -= augmented.get(i, j) * x[j];
                }
                x[i] /= augmented.get(i, i);
            }
            
            return x;
        }
        
        public static double[] jacobiIteration(Matrix A, double[] b, double[] x0, double tolerance, int maxIterations) {
            int n = A.getRows();
            double[] x = x0.clone();
            double[] xNew = new double[n];
            
            for (int iter = 0; iter < maxIterations; iter++) {
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            sum += A.get(i, j) * x[j];
                        }
                    }
                    xNew[i] = (b[i] - sum) / A.get(i, i);
                }
                
                // Check convergence
                double maxDiff = 0;
                for (int i = 0; i < n; i++) {
                    maxDiff = Math.max(maxDiff, Math.abs(xNew[i] - x[i]));
                }
                
                if (maxDiff < tolerance) {
                    return xNew;
                }
                
                x = xNew.clone();
            }
            
            return xNew;
        }
        
        public static double[] gaussSeidelIteration(Matrix A, double[] b, double[] x0, double tolerance, int maxIterations) {
            int n = A.getRows();
            double[] x = x0.clone();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] xOld = x.clone();
                
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            sum += A.get(i, j) * x[j];
                        }
                    }
                    x[i] = (b[i] - sum) / A.get(i, i);
                }
                
                // Check convergence
                double maxDiff = 0;
                for (int i = 0; i < n; i++) {
                    maxDiff = Math.max(maxDiff, Math.abs(x[i] - xOld[i]));
                }
                
                if (maxDiff < tolerance) {
                    return x;
                }
            }
            
            return x;
        }
        
        public static class LUDecomposition {
            public final Matrix L, U;
            public final int[] permutation;
            
            public LUDecomposition(Matrix L, Matrix U, int[] permutation) {
                this.L = L;
                this.U = U;
                this.permutation = permutation;
            }
        }
        
        public static LUDecomposition luDecomposition(Matrix A) {
            int n = A.getRows();
            Matrix L = new Matrix(n, n);
            Matrix U = new Matrix(n, n);
            int[] perm = new int[n];
            
            // Initialize
            for (int i = 0; i < n; i++) {
                perm[i] = i;
                L.set(i, i, 1.0);
                for (int j = 0; j < n; j++) {
                    U.set(i, j, A.get(i, j));
                }
            }
            
            // Decomposition with partial pivoting
            for (int k = 0; k < n - 1; k++) {
                // Find pivot
                int maxRow = k;
                for (int i = k + 1; i < n; i++) {
                    if (Math.abs(U.get(i, k)) > Math.abs(U.get(maxRow, k))) {
                        maxRow = i;
                    }
                }
                
                // Swap rows in U and permutation
                if (maxRow != k) {
                    for (int j = 0; j < n; j++) {
                        double temp = U.get(k, j);
                        U.set(k, j, U.get(maxRow, j));
                        U.set(maxRow, j, temp);
                    }
                    
                    int tempPerm = perm[k];
                    perm[k] = perm[maxRow];
                    perm[maxRow] = tempPerm;
                    
                    // Swap corresponding rows in L
                    for (int j = 0; j < k; j++) {
                        double temp = L.get(k, j);
                        L.set(k, j, L.get(maxRow, j));
                        L.set(maxRow, j, temp);
                    }
                }
                
                // Elimination
                for (int i = k + 1; i < n; i++) {
                    double factor = U.get(i, k) / U.get(k, k);
                    L.set(i, k, factor);
                    
                    for (int j = k; j < n; j++) {
                        U.set(i, j, U.get(i, j) - factor * U.get(k, j));
                    }
                }
            }
            
            return new LUDecomposition(L, U, perm);
        }
    }
    
    /**
     * Interpolation and Approximation Methods
     */
    public static class Interpolation {
        
        public static class PolynomialInterpolation {
            private final double[] coefficients;
            private final int degree;
            
            public PolynomialInterpolation(double[] coefficients) {
                this.coefficients = coefficients.clone();
                this.degree = coefficients.length - 1;
            }
            
            public double evaluate(double x) {
                double result = 0;
                double xPower = 1;
                
                for (int i = 0; i <= degree; i++) {
                    result += coefficients[i] * xPower;
                    xPower *= x;
                }
                
                return result;
            }
            
            public double[] getCoefficients() { return coefficients.clone(); }
            public int getDegree() { return degree; }
        }
        
        public static PolynomialInterpolation lagrangeInterpolation(double[] x, double[] y) {
            int n = x.length;
            double[] coefficients = new double[n];
            
            for (int i = 0; i < n; i++) {
                // Compute Lagrange basis polynomial coefficients
                double[] basisCoeffs = new double[n];
                basisCoeffs[0] = 1.0;
                
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        // Multiply by (x - x[j]) / (x[i] - x[j])
                        double[] newCoeffs = new double[n];
                        double denominator = x[i] - x[j];
                        
                        for (int k = 0; k < n - 1; k++) {
                            newCoeffs[k] -= basisCoeffs[k] * x[j] / denominator;
                            newCoeffs[k + 1] += basisCoeffs[k] / denominator;
                        }
                        
                        basisCoeffs = newCoeffs;
                    }
                }
                
                // Add to final polynomial
                for (int k = 0; k < n; k++) {
                    coefficients[k] += y[i] * basisCoeffs[k];
                }
            }
            
            return new PolynomialInterpolation(coefficients);
        }
        
        public static PolynomialInterpolation newtonInterpolation(double[] x, double[] y) {
            int n = x.length;
            double[][] dividedDiff = new double[n][n];
            
            // Initialize first column
            for (int i = 0; i < n; i++) {
                dividedDiff[i][0] = y[i];
            }
            
            // Compute divided differences
            for (int j = 1; j < n; j++) {
                for (int i = 0; i < n - j; i++) {
                    dividedDiff[i][j] = (dividedDiff[i + 1][j - 1] - dividedDiff[i][j - 1]) / (x[i + j] - x[i]);
                }
            }
            
            // Build polynomial coefficients
            double[] coefficients = new double[n];
            coefficients[0] = dividedDiff[0][0];
            
            for (int i = 1; i < n; i++) {
                // Coefficient of Newton basis polynomial
                double coeff = dividedDiff[0][i];
                
                // Convert to standard polynomial form
                double[] newCoeffs = new double[n];
                newCoeffs[0] = coeff;
                
                for (int j = 0; j < i; j++) {
                    double[] tempCoeffs = new double[n];
                    for (int k = 0; k < n - 1; k++) {
                        tempCoeffs[k] -= newCoeffs[k] * x[j];
                        tempCoeffs[k + 1] += newCoeffs[k];
                    }
                    newCoeffs = tempCoeffs;
                }
                
                for (int k = 0; k < n; k++) {
                    coefficients[k] += newCoeffs[k];
                }
            }
            
            return new PolynomialInterpolation(coefficients);
        }
        
        public static class SplineInterpolation {
            private final double[] x, y;
            private final double[] a, b, c, d; // Cubic spline coefficients
            
            public SplineInterpolation(double[] x, double[] y) {
                this.x = x.clone();
                this.y = y.clone();
                int n = x.length;
                
                // Compute cubic spline coefficients
                this.a = new double[n];
                this.b = new double[n];
                this.c = new double[n];
                this.d = new double[n];
                
                // Natural spline conditions
                double[] h = new double[n - 1];
                for (int i = 0; i < n - 1; i++) {
                    h[i] = x[i + 1] - x[i];
                }
                
                // Solve tridiagonal system for second derivatives
                double[] alpha = new double[n - 1];
                for (int i = 1; i < n - 1; i++) {
                    alpha[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
                }
                
                double[] l = new double[n];
                double[] mu = new double[n];
                double[] z = new double[n];
                
                l[0] = 1;
                for (int i = 1; i < n - 1; i++) {
                    l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
                    mu[i] = h[i] / l[i];
                    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
                }
                
                l[n - 1] = 1;
                z[n - 1] = 0;
                c[n - 1] = 0;
                
                for (int j = n - 2; j >= 0; j--) {
                    c[j] = z[j] - mu[j] * c[j + 1];
                    b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
                    d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
                    a[j] = y[j];
                }
            }
            
            public double evaluate(double t) {
                // Find interval
                int i = 0;
                for (int j = 0; j < x.length - 1; j++) {
                    if (t >= x[j] && t <= x[j + 1]) {
                        i = j;
                        break;
                    }
                }
                
                double dt = t - x[i];
                return a[i] + b[i] * dt + c[i] * dt * dt + d[i] * dt * dt * dt;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Numerical Methods Algorithms Demo:");
        System.out.println("=================================");
        
        // Root Finding demonstration
        System.out.println("1. Root Finding Methods:");
        Function<Double, Double> f = x -> x * x * x - 2 * x - 5; // f(x) = x³ - 2x - 5
        Function<Double, Double> df = x -> 3 * x * x - 2; // f'(x) = 3x² - 2
        
        RootFinding.RootResult bisection = RootFinding.bisectionMethod(f, 2.0, 3.0, 1e-6, 100);
        System.out.println("Bisection: " + bisection);
        
        RootFinding.RootResult newton = RootFinding.newtonRaphsonMethod(f, df, 2.5, 1e-6, 100);
        System.out.println("Newton-Raphson: " + newton);
        
        RootFinding.RootResult secant = RootFinding.secantMethod(f, 2.0, 3.0, 1e-6, 100);
        System.out.println("Secant: " + secant);
        
        // Numerical Integration
        System.out.println("\n2. Numerical Integration:");
        Function<Double, Double> integrand = x -> Math.exp(-x * x); // e^(-x²)
        
        NumericalIntegration.IntegrationResult trapezoidal = 
            NumericalIntegration.trapezoidalRule(integrand, 0, 2, 100);
        System.out.println("Trapezoidal rule: " + trapezoidal);
        
        NumericalIntegration.IntegrationResult simpsons = 
            NumericalIntegration.simpsonsRule(integrand, 0, 2, 100);
        System.out.println("Simpson's rule: " + simpsons);
        
        NumericalIntegration.IntegrationResult gauss = 
            NumericalIntegration.gaussianQuadrature(integrand, 0, 2, 4);
        System.out.println("Gaussian quadrature: " + gauss);
        
        NumericalIntegration.IntegrationResult monteCarlo = 
            NumericalIntegration.monteCarloIntegration(integrand, 0, 2, 10000);
        System.out.println("Monte Carlo: " + monteCarlo);
        
        // ODE Solving
        System.out.println("\n3. Ordinary Differential Equations:");
        BiFunction<Double, Double, Double> ode = (x, y) -> -2 * x * y; // dy/dx = -2xy
        
        ODESolvers.ODEResult euler = ODESolvers.eulerMethod(ode, 0, 1, 1, 0.1);
        System.out.println("Euler method: " + euler);
        
        ODESolvers.ODEResult rk4 = ODESolvers.rungeKutta4(ode, 0, 1, 1, 0.1);
        System.out.println("Runge-Kutta 4: " + rk4);
        
        // System of ODEs (Lotka-Volterra predator-prey model)
        Function<double[], double[]> predatorPrey = state -> {
            double x = state[0]; // prey
            double y = state[1]; // predator
            double alpha = 1.0, beta = 0.5, gamma = 0.5, delta = 1.0;
            
            return new double[]{
                alpha * x - beta * x * y,  // dx/dt
                delta * x * y - gamma * y  // dy/dt
            };
        };
        
        ODESolvers.SystemODEResult system = ODESolvers.rungeKutta4System(
            predatorPrey, 0, new double[]{10, 5}, 10, 0.01);
        System.out.printf("Predator-prey system: %d steps, final prey=%.3f, predator=%.3f\n",
                         system.steps, system.y[0][system.y[0].length-1], system.y[1][system.y[1].length-1]);
        
        // Linear Algebra
        System.out.println("\n4. Linear Algebra:");
        LinearAlgebra.Matrix A = new LinearAlgebra.Matrix(new double[][]{
            {3, 2, -1},
            {2, -2, 4},
            {-1, 0.5, -1}
        });
        double[] b = {1, -2, 0};
        
        double[] solution = LinearAlgebra.gaussianElimination(A, b);
        System.out.printf("Gaussian elimination solution: [%.3f, %.3f, %.3f]\n", 
                         solution[0], solution[1], solution[2]);
        
        double[] jacobi = LinearAlgebra.jacobiIteration(A, b, new double[]{0, 0, 0}, 1e-6, 100);
        System.out.printf("Jacobi iteration solution: [%.3f, %.3f, %.3f]\n", 
                         jacobi[0], jacobi[1], jacobi[2]);
        
        System.out.printf("Matrix determinant: %.3f\n", A.determinant());
        
        // LU Decomposition
        LinearAlgebra.LUDecomposition lu = LinearAlgebra.luDecomposition(A);
        System.out.println("LU decomposition completed");
        
        // Interpolation
        System.out.println("\n5. Interpolation:");
        double[] xPoints = {0, 1, 2, 3, 4};
        double[] yPoints = {1, 2.718, 7.389, 20.086, 54.598}; // Approximate e^x
        
        Interpolation.PolynomialInterpolation lagrange = 
            Interpolation.lagrangeInterpolation(xPoints, yPoints);
        System.out.printf("Lagrange interpolation at x=2.5: %.3f\n", lagrange.evaluate(2.5));
        
        Interpolation.PolynomialInterpolation newton = 
            Interpolation.newtonInterpolation(xPoints, yPoints);
        System.out.printf("Newton interpolation at x=2.5: %.3f\n", newton.evaluate(2.5));
        
        Interpolation.SplineInterpolation spline = 
            new Interpolation.SplineInterpolation(xPoints, yPoints);
        System.out.printf("Cubic spline interpolation at x=2.5: %.3f\n", spline.evaluate(2.5));
        
        System.out.println("\nNumerical methods demonstration completed!");
        System.out.println("Methods demonstrated:");
        System.out.println("- Root finding: Bisection, Newton-Raphson, Secant, False Position");
        System.out.println("- Integration: Trapezoidal, Simpson's, Gaussian Quadrature, Monte Carlo");
        System.out.println("- ODE solving: Euler, Heun, Runge-Kutta 4, Adams-Bashforth");
        System.out.println("- Linear systems: Gaussian elimination, Jacobi, Gauss-Seidel, LU decomposition");
        System.out.println("- Interpolation: Lagrange, Newton, Cubic splines");
    }
}
