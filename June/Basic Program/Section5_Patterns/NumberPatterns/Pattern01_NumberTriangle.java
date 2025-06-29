import java.util.Scanner;

public class Pattern01_NumberTriangle {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Number Triangle Pattern:");
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(j + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
