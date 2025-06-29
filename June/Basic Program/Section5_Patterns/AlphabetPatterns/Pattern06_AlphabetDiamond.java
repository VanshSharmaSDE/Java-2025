import java.util.Scanner;

public class Pattern06_AlphabetDiamond {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows (for upper half): ");
        int n = scanner.nextInt();
        
        System.out.println("Alphabet Diamond Pattern:");
        
        // Upper half
        for (int i = 1; i <= n; i++) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print("  ");
            }
            // Print ascending alphabets
            char ch = 'A';
            for (int j = 1; j <= i; j++) {
                System.out.print(ch + " ");
                ch++;
            }
            // Print descending alphabets
            ch -= 2;
            for (int j = 1; j <= i - 1; j++) {
                System.out.print(ch + " ");
                ch--;
            }
            System.out.println();
        }
        
        // Lower half
        for (int i = n - 1; i >= 1; i--) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print("  ");
            }
            // Print ascending alphabets
            char ch = 'A';
            for (int j = 1; j <= i; j++) {
                System.out.print(ch + " ");
                ch++;
            }
            // Print descending alphabets
            ch -= 2;
            for (int j = 1; j <= i - 1; j++) {
                System.out.print(ch + " ");
                ch--;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
