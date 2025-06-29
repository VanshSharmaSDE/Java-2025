import java.util.Scanner;

public class SumOfFirstNNaturalNumbers {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of N: ");
        int n = scanner.nextInt();
        
        // Method 1: Using loop
        int sum1 = 0;
        for (int i = 1; i <= n; i++) {
            sum1 += i;
        }
        
        // Method 2: Using formula n*(n+1)/2
        int sum2 = n * (n + 1) / 2;
        
        System.out.println("Sum using loop: " + sum1);
        System.out.println("Sum using formula: " + sum2);
        
        scanner.close();
    }
}
