package Algorithms.Mathematics.Calculus;

/**
 * Numerical Calculus Algorithms
 */
public class NumericalCalculus {
    
    /**
     * Numerical differentiation using forward difference
     * f'(x) ≈ (f(x+h) - f(x))/h
     * @param x Point at which to calculate derivative
     * @param h Step size
     * @return Approximate derivative
     */
    public static double forwardDifference(double x, double h) {
        return (function(x + h) - function(x)) / h;
    }
    
    /**
     * Numerical differentiation using central difference
     * f'(x) ≈ (f(x+h) - f(x-h))/(2h)
     * @param x Point at which to calculate derivative
     * @param h Step size
     * @return Approximate derivative
     */
    public static double centralDifference(double x, double h) {
        return (function(x + h) - function(x - h)) / (2 * h);
    }
    
    /**
     * Numerical integration using Trapezoidal Rule
     * ∫[a,b] f(x)dx ≈ (b-a)/2 * (f(a) + f(b))
     * @param a Lower limit
     * @param b Upper limit
     * @param n Number of intervals
     * @return Approximate integral
     */
    public static double trapezoidalRule(double a, double b, int n) {
        double h = (b - a) / n;
        double sum = function(a) + function(b);
        
        for (int i = 1; i < n; i++) {
            sum += 2 * function(a + i * h);
        }
        
        return (h / 2) * sum;
    }
    
    /**
     * Numerical integration using Simpson's Rule
     * ∫[a,b] f(x)dx ≈ (b-a)/6 * (f(a) + 4f((a+b)/2) + f(b))
     * @param a Lower limit
     * @param b Upper limit
     * @param n Number of intervals (must be even)
     * @return Approximate integral
     */
    public static double simpsonsRule(double a, double b, int n) {
        if (n % 2 != 0) {
            throw new IllegalArgumentException("Number of intervals must be even for Simpson's rule");
        }
        
        double h = (b - a) / n;
        double sum = function(a) + function(b);
        
        for (int i = 1; i < n; i++) {
            double x = a + i * h;
            if (i % 2 == 0) {
                sum += 2 * function(x);
            } else {
                sum += 4 * function(x);
            }
        }
        
        return (h / 3) * sum;
    }
    
    /**
     * Newton-Raphson method for finding roots
     * x_{n+1} = x_n - f(x_n)/f'(x_n)
     * @param initialGuess Initial guess for root
     * @param tolerance Tolerance for convergence
     * @param maxIterations Maximum number of iterations
     * @return Approximate root
     */
    public static double newtonRaphson(double initialGuess, double tolerance, int maxIterations) {
        double x = initialGuess;
        
        for (int i = 0; i < maxIterations; i++) {
            double fx = function(x);
            double fpx = centralDifference(x, 1e-8);
            
            if (Math.abs(fpx) < 1e-12) {
                throw new RuntimeException("Derivative too small, method may not converge");
            }
            
            double newX = x - fx / fpx;
            
            if (Math.abs(newX - x) < tolerance) {
                return newX;
            }
            
            x = newX;
        }
        
        throw new RuntimeException("Method did not converge within " + maxIterations + " iterations");
    }
    
    /**
     * Taylor series expansion for e^x
     * e^x = 1 + x + x²/2! + x³/3! + ...
     * @param x Input value
     * @param terms Number of terms to include
     * @return Approximate value of e^x
     */
    public static double taylorSeriesExp(double x, int terms) {
        double result = 1.0;
        double term = 1.0;
        
        for (int i = 1; i < terms; i++) {
            term *= x / i;
            result += term;
        }
        
        return result;
    }
    
    /**
     * Taylor series expansion for sin(x)
     * sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
     * @param x Input value in radians
     * @param terms Number of terms to include
     * @return Approximate value of sin(x)
     */
    public static double taylorSeriesSin(double x, int terms) {
        double result = 0.0;
        
        for (int i = 0; i < terms; i++) {
            int power = 2 * i + 1;
            double term = Math.pow(x, power) / factorial(power);
            if (i % 2 == 0) {
                result += term;
            } else {
                result -= term;
            }
        }
        
        return result;
    }
    
    /**
     * Example function: f(x) = x² - 4
     * @param x Input value
     * @return Function value
     */
    private static double function(double x) {
        return x * x - 4; // Example: f(x) = x² - 4
    }
    
    /**
     * Calculate factorial
     * @param n Input number
     * @return n!
     */
    private static double factorial(int n) {
        if (n <= 1) return 1;
        double result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    
    public static void main(String[] args) {
        System.out.println("Numerical Calculus Algorithms:");
        System.out.println("==============================");
        
        double x = 2.0;
        double h = 1e-6;
        
        // Numerical differentiation
        System.out.println("Function: f(x) = x² - 4");
        System.out.println("At x = " + x + ":");
        System.out.println("Forward difference f'(x) ≈ " + forwardDifference(x, h));
        System.out.println("Central difference f'(x) ≈ " + centralDifference(x, h));
        System.out.println("Analytical derivative f'(x) = 2x = " + (2 * x));
        
        // Numerical integration
        double a = 0, b = 3;
        int n = 1000;
        System.out.println("\nNumerical Integration from " + a + " to " + b + ":");
        System.out.println("Trapezoidal rule: " + trapezoidalRule(a, b, n));
        System.out.println("Simpson's rule: " + simpsonsRule(a, b, n));
        
        // Root finding
        System.out.println("\nRoot finding for f(x) = x² - 4:");
        try {
            double root = newtonRaphson(1.5, 1e-10, 100);
            System.out.println("Root found: " + root);
            System.out.println("Verification f(" + root + ") = " + function(root));
        } catch (RuntimeException e) {
            System.out.println("Error: " + e.getMessage());
        }
        
        // Taylor series
        double value = 1.0;
        System.out.println("\nTaylor Series at x = " + value + ":");
        System.out.println("e^x (10 terms): " + taylorSeriesExp(value, 10));
        System.out.println("Math.exp(x): " + Math.exp(value));
        System.out.println("sin(x) (10 terms): " + taylorSeriesSin(value, 10));
        System.out.println("Math.sin(x): " + Math.sin(value));
    }
}
