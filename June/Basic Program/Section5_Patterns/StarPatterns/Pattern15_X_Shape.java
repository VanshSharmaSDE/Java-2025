import java.util.Scanner;

public class Pattern15_X_Shape {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size (odd number): ");
        int n = scanner.nextInt();
        
        System.out.println("X Shape Pattern:");
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == j || i + j == n + 1) {
                    System.out.print("* ");
                } else {
                    System.out.print("  ");
                }
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
