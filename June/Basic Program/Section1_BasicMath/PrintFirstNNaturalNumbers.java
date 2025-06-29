import java.util.Scanner;

public class PrintFirstNNaturalNumbers {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of N: ");
        int n = scanner.nextInt();
        
        System.out.println("First " + n + " natural numbers:");
        for (int i = 1; i <= n; i++) {
            System.out.print(i + " ");
        }
        System.out.println();
        
        scanner.close();
    }
}
