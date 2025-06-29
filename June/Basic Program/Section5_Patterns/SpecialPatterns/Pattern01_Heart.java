import java.util.Scanner;

public class Pattern01_Heart {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size (recommended 6-10): ");
        int n = scanner.nextInt();
        
        System.out.println("Heart Pattern:");
        
        // Upper part of heart
        for (int i = n / 2; i <= n; i += 2) {
            // Left spaces
            for (int j = 1; j < n - i / 2; j++) {
                System.out.print(" ");
            }
            // Left stars
            for (int j = 1; j <= i; j++) {
                System.out.print("*");
            }
            // Middle spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print(" ");
            }
            // Right stars
            for (int j = 1; j <= i; j++) {
                System.out.print("*");
            }
            System.out.println();
        }
        
        // Lower part of heart (inverted triangle)
        for (int i = n; i >= 1; i--) {
            for (int j = i; j < n; j++) {
                System.out.print(" ");
            }
            for (int j = 1; j <= (i * 2) - 1; j++) {
                System.out.print("*");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
