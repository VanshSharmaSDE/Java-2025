import java.util.Scanner;

public class Pattern01_Square {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size: ");
        int n = scanner.nextInt();
        
        System.out.println("Square Pattern:");
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                System.out.print("* ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
