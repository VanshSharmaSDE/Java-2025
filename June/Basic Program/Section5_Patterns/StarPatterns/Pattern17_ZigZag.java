import java.util.Scanner;

public class Pattern17_ZigZag {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int rows = scanner.nextInt();
        System.out.print("Enter the number of columns: ");
        int cols = scanner.nextInt();
        
        System.out.println("ZigZag Pattern:");
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                if (((i + j) % 4 == 0) || (i == 2 && j % 4 == 0)) {
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
