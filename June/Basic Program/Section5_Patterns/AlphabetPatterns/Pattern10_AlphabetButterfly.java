import java.util.Scanner;

public class Pattern10_AlphabetButterfly {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Alphabet Butterfly Pattern:");
        
        // Upper half
        for (int i = 1; i <= n; i++) {
            // Left alphabets
            char ch = 'A';
            for (int j = 1; j <= i; j++) {
                System.out.print(ch);
                ch++;
            }
            // Spaces in middle
            for (int j = 1; j <= 2 * (n - i); j++) {
                System.out.print(" ");
            }
            // Right alphabets
            ch = (char)('A' + i - 1);
            for (int j = 1; j <= i; j++) {
                System.out.print(ch);
                ch--;
            }
            System.out.println();
        }
        
        // Lower half
        for (int i = n - 1; i >= 1; i--) {
            // Left alphabets
            char ch = 'A';
            for (int j = 1; j <= i; j++) {
                System.out.print(ch);
                ch++;
            }
            // Spaces in middle
            for (int j = 1; j <= 2 * (n - i); j++) {
                System.out.print(" ");
            }
            // Right alphabets
            ch = (char)('A' + i - 1);
            for (int j = 1; j <= i; j++) {
                System.out.print(ch);
                ch--;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
