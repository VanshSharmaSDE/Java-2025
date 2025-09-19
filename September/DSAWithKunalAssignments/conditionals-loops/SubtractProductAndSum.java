import java.util.Scanner;

public class SubtractProductAndSum {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = sc.nextInt();
        
        int originalNumber = number;
        int sum = 0;
        int product = 1;
        
        // Calculate sum and product of digits
        while (number > 0) {
            int digit = number % 10;
            sum += digit;
            product *= digit;
            number /= 10;
        }
        
        int result = product - sum;
        
        System.out.println("Number: " + originalNumber);
        System.out.println("Sum of digits: " + sum);
        System.out.println("Product of digits: " + product);
        System.out.println("Product - Sum = " + result);
        
        sc.close();
    }
}