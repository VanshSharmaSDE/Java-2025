import java.util.Scanner;

public class SumOfDigitsRecursive {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        int sum = sumOfDigits(Math.abs(number)); // Use absolute value for negative numbers
        
        System.out.println("Sum of digits of " + number + " is: " + sum);
        
        scanner.close();
    }
    
    public static int sumOfDigits(int n) {
        // Base case
        if (n == 0) {
            return 0;
        }
        
        // Recursive case: last digit + sum of remaining digits
        return (n % 10) + sumOfDigits(n / 10);
    }
}
