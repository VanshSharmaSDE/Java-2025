import java.util.Scanner;

public class Pattern03_RowAlphabetTriangle {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Row Alphabet Triangle Pattern:");
        for (int i = 1; i <= n; i++) {
            char ch = (char)('A' + i - 1);
            for (int j = 1; j <= i; j++) {
                System.out.print(ch + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
