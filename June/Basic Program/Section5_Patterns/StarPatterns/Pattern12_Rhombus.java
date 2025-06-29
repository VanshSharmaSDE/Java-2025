import java.util.Scanner;

public class Pattern12_Rhombus {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Rhombus Pattern:");
        for (int i = 1; i <= n; i++) {
            // Print spaces
            for (int j = 1; j <= n - i; j++) {
                System.out.print(" ");
            }
            // Print stars
            for (int j = 1; j <= n; j++) {
                System.out.print("* ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
