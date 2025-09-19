import java.util.Scanner;

public class AdditionOfTwoNumbers {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter first number: ");
        double num1 = sc.nextDouble();
        
        System.out.print("Enter second number: ");
        double num2 = sc.nextDouble();
        
        double sum = num1 + num2;
        
        System.out.println("Sum of " + num1 + " and " + num2 + " is: " + sum);
        
        sc.close();
    }
}