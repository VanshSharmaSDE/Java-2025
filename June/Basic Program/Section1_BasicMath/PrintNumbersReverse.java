import java.util.Scanner;

public class PrintNumbersReverse {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of N: ");
        int n = scanner.nextInt();
        
        System.out.println("Numbers from " + n + " to 1:");
        for (int i = n; i >= 1; i--) {
            System.out.print(i + " ");
        }
        System.out.println();
        
        scanner.close();
    }
}
