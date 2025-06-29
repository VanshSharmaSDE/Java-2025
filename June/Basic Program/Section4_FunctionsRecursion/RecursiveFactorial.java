import java.util.Scanner;

public class RecursiveFactorial {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        if (number < 0) {
            System.out.println("Factorial is not defined for negative numbers.");
        } else {
            long result = factorial(number);
            System.out.println("Factorial of " + number + " is: " + result);
        }
        
        scanner.close();
    }
    
    public static long factorial(int n) {
        // Base case
        if (n == 0 || n == 1) {
            return 1;
        }
        
        // Recursive case
        return n * factorial(n - 1);
    }
}
