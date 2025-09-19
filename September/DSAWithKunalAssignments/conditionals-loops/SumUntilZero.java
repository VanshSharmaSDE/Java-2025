import java.util.Scanner;

public class SumUntilZero {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int sum = 0;
        int number;
        
        System.out.println("Enter numbers (enter 0 to stop):");
        
        while (true) {
            System.out.print("Enter a number: ");
            number = sc.nextInt();
            
            if (number == 0) {
                break;
            }
            
            sum += number;
        }
        
        System.out.println("Sum of all numbers: " + sum);
        
        sc.close();
    }
}