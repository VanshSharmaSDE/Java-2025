import java.util.Scanner;

public class Pattern06_NumberPyramid {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Number Pyramid Pattern:");
        for (int i = 1; i <= n; i++) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print("  ");
            }
            // Print numbers
            for (int j = 1; j <= i; j++) {
                System.out.print(j + " ");
            }
            // Print reverse numbers
            for (int j = i - 1; j >= 1; j--) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
