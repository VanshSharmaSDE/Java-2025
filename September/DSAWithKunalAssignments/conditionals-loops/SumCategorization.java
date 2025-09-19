import java.util.Scanner;

public class SumCategorization {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int sumNegative = 0;
        int sumPositiveEven = 0;
        int sumPositiveOdd = 0;
        int number;
        
        System.out.println("Enter numbers (enter 0 to terminate):");
        
        while (true) {
            System.out.print("Enter a number: ");
            number = sc.nextInt();
            
            if (number == 0) {
                break;
            }
            
            if (number < 0) {
                sumNegative += number;
            } else { // number > 0
                if (number % 2 == 0) {
                    sumPositiveEven += number;
                } else {
                    sumPositiveOdd += number;
                }
            }
        }
        
        System.out.println("\nResults:");
        System.out.println("Sum of negative numbers: " + sumNegative);
        System.out.println("Sum of positive even numbers: " + sumPositiveEven);
        System.out.println("Sum of positive odd numbers: " + sumPositiveOdd);
        
        sc.close();
    }
}