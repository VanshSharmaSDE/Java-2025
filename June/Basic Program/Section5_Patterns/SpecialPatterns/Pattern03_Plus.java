import java.util.Scanner;

public class Pattern03_Plus {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size (odd number): ");
        int n = scanner.nextInt();
        
        System.out.println("Plus Pattern:");
        int mid = n / 2 + 1;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == mid || j == mid) {
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
