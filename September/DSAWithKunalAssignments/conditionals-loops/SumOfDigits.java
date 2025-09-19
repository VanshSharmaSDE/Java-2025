import java.util.Scanner;

public class SumOfDigits {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = sc.nextInt();
        
        int originalNumber = number;
        int sum = 0;
        
        while (number > 0) {
            int digit = number % 10;
            sum += digit;
            number /= 10;
        }
        
        System.out.println("Sum of digits of " + originalNumber + " is: " + sum);
        
        sc.close();
    }
}