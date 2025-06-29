import java.util.Scanner;

public class Pattern05_Swastika {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size (odd number >= 7): ");
        int n = scanner.nextInt();
        
        System.out.println("Swastika Pattern:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < n / 2) {
                    if (j < n / 2) {
                        if (j == 0) System.out.print("* ");
                        else System.out.print("  ");
                    } else if (j == n / 2) {
                        System.out.print("* ");
                    } else {
                        if (i == 0) System.out.print("* ");
                        else System.out.print("  ");
                    }
                } else if (i == n / 2) {
                    System.out.print("* ");
                } else {
                    if (j < n / 2) {
                        if (i == n - 1) System.out.print("* ");
                        else System.out.print("  ");
                    } else if (j == n / 2) {
                        System.out.print("* ");
                    } else {
                        if (j == n - 1) System.out.print("* ");
                        else System.out.print("  ");
                    }
                }
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
