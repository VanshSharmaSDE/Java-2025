import java.util.Scanner;

public class LargestNumberUntilZero {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int largest = Integer.MIN_VALUE;
        int number;
        boolean hasInput = false;
        
        System.out.println("Enter numbers (enter 0 to stop):");
        
        while (true) {
            System.out.print("Enter a number: ");
            number = sc.nextInt();
            
            if (number == 0) {
                break;
            }
            
            hasInput = true;
            if (number > largest) {
                largest = number;
            }
        }
        
        if (hasInput) {
            System.out.println("Largest number: " + largest);
        } else {
            System.out.println("No numbers were entered.");
        }
        
        sc.close();
    }
}