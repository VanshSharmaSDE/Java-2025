import java.util.Scanner;

public class Pattern11_EvenNumberTriangle {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Even Number Triangle Pattern:");
        for (int i = 1; i <= n; i++) {
            int evenNum = 2;
            for (int j = 1; j <= i; j++) {
                System.out.print(evenNum + " ");
                evenNum += 2;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
