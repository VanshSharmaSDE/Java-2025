import java.util.Scanner;

public class Pattern15_TablePattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number: ");
        int num = scanner.nextInt();
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Table Pattern for " + num + ":");
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print((num * j) + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
