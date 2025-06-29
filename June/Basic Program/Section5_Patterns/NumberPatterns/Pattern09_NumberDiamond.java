import java.util.Scanner;

public class Pattern09_NumberDiamond {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows (for upper half): ");
        int n = scanner.nextInt();
        
        System.out.println("Number Diamond Pattern:");
        
        // Upper half
        for (int i = 1; i <= n; i++) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print("  ");
            }
            // Print ascending numbers
            for (int j = 1; j <= i; j++) {
                System.out.print(j + " ");
            }
            // Print descending numbers
            for (int j = i - 1; j >= 1; j--) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
        
        // Lower half
        for (int i = n - 1; i >= 1; i--) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print("  ");
            }
            // Print ascending numbers
            for (int j = 1; j <= i; j++) {
                System.out.print(j + " ");
            }
            // Print descending numbers
            for (int j = i - 1; j >= 1; j--) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
