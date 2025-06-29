import java.util.Scanner;

public class Factorial {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        long iterativeResult = factorialIterative(number);
        long recursiveResult = factorialRecursive(number);
        
        System.out.println("Factorial using iterative method: " + iterativeResult);
        System.out.println("Factorial using recursive method: " + recursiveResult);
        
        scanner.close();
    }
    
    // Iterative method
    public static long factorialIterative(int n) {
        if (n < 0) {
            return -1; // Factorial is not defined for negative numbers
        }
        
        long factorial = 1;
        for (int i = 1; i <= n; i++) {
            factorial *= i;
        }
        return factorial;
    }
    
    // Recursive method
    public static long factorialRecursive(int n) {
        if (n < 0) {
            return -1; // Factorial is not defined for negative numbers
        }
        if (n == 0 || n == 1) {
            return 1;
        }
        return n * factorialRecursive(n - 1);
    }
}
