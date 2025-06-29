import java.util.Scanner;

public class Pattern12_OddNumberTriangle {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Odd Number Triangle Pattern:");
        for (int i = 1; i <= n; i++) {
            int oddNum = 1;
            for (int j = 1; j <= i; j++) {
                System.out.print(oddNum + " ");
                oddNum += 2;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
