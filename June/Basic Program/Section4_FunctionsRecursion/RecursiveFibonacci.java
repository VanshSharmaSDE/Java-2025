import java.util.Scanner;

public class RecursiveFibonacci {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of terms: ");
        int n = scanner.nextInt();
        
        if (n <= 0) {
            System.out.println("Please enter a positive number.");
        } else {
            System.out.println("Fibonacci series up to " + n + " terms:");
            for (int i = 0; i < n; i++) {
                System.out.print(fibonacci(i) + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
    
    public static int fibonacci(int n) {
        // Base cases
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        
        // Recursive case
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
