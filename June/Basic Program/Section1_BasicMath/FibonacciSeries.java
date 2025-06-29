import java.util.Scanner;

public class FibonacciSeries {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of terms: ");
        int n = scanner.nextInt();
        
        System.out.println("Fibonacci series up to " + n + " terms:");
        printFibonacci(n);
        
        scanner.close();
    }
    
    public static void printFibonacci(int n) {
        if (n <= 0) {
            System.out.println("Please enter a positive number.");
            return;
        }
        
        int first = 0, second = 1;
        
        if (n >= 1) {
            System.out.print(first + " ");
        }
        if (n >= 2) {
            System.out.print(second + " ");
        }
        
        for (int i = 3; i <= n; i++) {
            int next = first + second;
            System.out.print(next + " ");
            first = second;
            second = next;
        }
        System.out.println();
    }
}
