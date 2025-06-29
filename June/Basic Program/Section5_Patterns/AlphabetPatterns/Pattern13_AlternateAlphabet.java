import java.util.Scanner;

public class Pattern13_AlternateAlphabet {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Alternate Alphabet Pattern:");
        for (int i = 1; i <= n; i++) {
            char ch = 'A';
            for (int j = 1; j <= i; j++) {
                System.out.print(ch + " ");
                ch += 2; // Skip one letter
                if (ch > 'Z') {
                    ch = 'A';
                }
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
