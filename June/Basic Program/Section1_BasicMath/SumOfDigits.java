import java.util.Scanner;

public class SumOfDigits {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        int sum = sumOfDigits(number);
        
        System.out.println("Sum of digits of " + number + " is: " + sum);
        
        scanner.close();
    }
    
    public static int sumOfDigits(int n) {
        n = Math.abs(n); // Handle negative numbers
        int sum = 0;
        
        while (n > 0) {
            sum += n % 10;
            n /= 10;
        }
        
        return sum;
    }
}
