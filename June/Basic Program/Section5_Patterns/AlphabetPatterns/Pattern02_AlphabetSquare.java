import java.util.Scanner;

public class Pattern02_AlphabetSquare {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size: ");
        int n = scanner.nextInt();
        
        System.out.println("Alphabet Square Pattern:");
        for (int i = 1; i <= n; i++) {
            char ch = 'A';
            for (int j = 1; j <= n; j++) {
                System.out.print(ch + " ");
                ch++;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
