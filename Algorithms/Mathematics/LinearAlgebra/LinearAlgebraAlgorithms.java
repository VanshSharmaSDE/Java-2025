package Algorithms.Mathematics.LinearAlgebra;

/**
 * Linear Algebra Algorithms
 */
public class LinearAlgebraAlgorithms {
    
    /**
     * Matrix multiplication: C = A × B
     * @param A First matrix (m × n)
     * @param B Second matrix (n × p)
     * @return Product matrix C (m × p)
     */
    public static double[][] matrixMultiplication(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        if (n != B.length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        double[][] C = new double[m][p];
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
    
    /**
     * Matrix transpose
     * @param matrix Input matrix
     * @return Transposed matrix
     */
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        
        return transposed;
    }
    
    /**
     * Calculate determinant of a square matrix using recursion
     * @param matrix Square matrix
     * @return Determinant value
     */
    public static double determinant(double[][] matrix) {
        int n = matrix.length;
        
        if (n != matrix[0].length) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        
        if (n == 1) {
            return matrix[0][0];
        }
        
        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        
        double det = 0;
        for (int j = 0; j < n; j++) {
            double[][] minor = getMinor(matrix, 0, j);
            det += Math.pow(-1, j) * matrix[0][j] * determinant(minor);
        }
        
        return det;
    }
    
    /**
     * Get minor matrix by removing specified row and column
     * @param matrix Original matrix
     * @param excludeRow Row to exclude
     * @param excludeCol Column to exclude
     * @return Minor matrix
     */
    private static double[][] getMinor(double[][] matrix, int excludeRow, int excludeCol) {
        int n = matrix.length;
        double[][] minor = new double[n - 1][n - 1];
        
        int minorRow = 0;
        for (int i = 0; i < n; i++) {
            if (i == excludeRow) continue;
            
            int minorCol = 0;
            for (int j = 0; j < n; j++) {
                if (j == excludeCol) continue;
                
                minor[minorRow][minorCol] = matrix[i][j];
                minorCol++;
            }
            minorRow++;
        }
        
        return minor;
    }
    
    /**
     * Gaussian elimination to solve system of linear equations Ax = b
     * @param A Coefficient matrix
     * @param b Constants vector
     * @return Solution vector x
     */
    public static double[] gaussianElimination(double[][] A, double[] b) {
        int n = A.length;
        
        // Create augmented matrix
        double[][] augmented = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            // Make all rows below this one 0 in current column
            for (int k = i + 1; k < n; k++) {
                double factor = augmented[k][i] / augmented[i][i];
                for (int j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            x[i] /= augmented[i][i];
        }
        
        return x;
    }
    
    /**
     * Calculate dot product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Dot product
     */
    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * Calculate cross product of two 3D vectors
     * @param a First vector (3D)
     * @param b Second vector (3D)
     * @return Cross product vector
     */
    public static double[] crossProduct(double[] a, double[] b) {
        if (a.length != 3 || b.length != 3) {
            throw new IllegalArgumentException("Cross product requires 3D vectors");
        }
        
        return new double[] {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
    
    /**
     * Calculate vector magnitude (norm)
     * @param vector Input vector
     * @return Magnitude
     */
    public static double vectorMagnitude(double[] vector) {
        double sum = 0;
        for (double v : vector) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Print matrix in formatted way
     * @param matrix Matrix to print
     */
    public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double element : row) {
                System.out.printf("%8.2f ", element);
            }
            System.out.println();
        }
    }
    
    /**
     * Print vector
     * @param vector Vector to print
     */
    public static void printVector(double[] vector) {
        System.out.print("[");
        for (int i = 0; i < vector.length; i++) {
            System.out.printf("%.2f", vector[i]);
            if (i < vector.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
    
    public static void main(String[] args) {
        System.out.println("Linear Algebra Algorithms:");
        System.out.println("==========================");
        
        // Matrix operations
        double[][] A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        
        System.out.println("Matrix A:");
        printMatrix(A);
        System.out.println("\nMatrix B:");
        printMatrix(B);
        
        // Matrix multiplication
        double[][] C = matrixMultiplication(A, B);
        System.out.println("\nA × B:");
        printMatrix(C);
        
        // Transpose
        double[][] AT = transpose(A);
        System.out.println("\nTranspose of A:");
        printMatrix(AT);
        
        // Determinant
        double[][] detMatrix = {{4, 6}, {3, 8}};
        double det = determinant(detMatrix);
        System.out.println("\nDeterminant of 2x2 matrix:");
        printMatrix(detMatrix);
        System.out.println("Determinant: " + det);
        
        // System of linear equations: Ax = b
        double[][] coeffMatrix = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
        double[] constants = {8, -11, -3};
        
        System.out.println("\nSolving system Ax = b:");
        System.out.println("Coefficient matrix:");
        printMatrix(coeffMatrix);
        System.out.println("Constants:");
        printVector(constants);
        
        double[] solution = gaussianElimination(coeffMatrix, constants);
        System.out.println("Solution:");
        printVector(solution);
        
        // Vector operations
        double[] v1 = {1, 2, 3};
        double[] v2 = {4, 5, 6};
        
        System.out.println("\nVector operations:");
        System.out.print("v1 = ");
        printVector(v1);
        System.out.print("v2 = ");
        printVector(v2);
        
        double dot = dotProduct(v1, v2);
        System.out.println("Dot product: " + dot);
        
        double[] cross = crossProduct(v1, v2);
        System.out.print("Cross product: ");
        printVector(cross);
        
        double mag1 = vectorMagnitude(v1);
        System.out.println("Magnitude of v1: " + mag1);
    }
}
