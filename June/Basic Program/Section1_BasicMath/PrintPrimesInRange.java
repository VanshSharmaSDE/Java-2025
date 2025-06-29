import java.util.Scanner;

public class PrintPrimesInRange {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the start of range: ");
        int start = scanner.nextInt();
        System.out.print("Enter the end of range: ");
        int end = scanner.nextInt();
        
        System.out.println("Prime numbers between " + start + " and " + end + ":");
        for (int i = start; i <= end; i++) {
            if (isPrime(i)) {
                System.out.print(i + " ");
            }
        }
        System.out.println();
        
        scanner.close();
    }
    
    public static boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        if (n <= 3) {
            return true;
        }
        if (n % 2 == 0 || n % 3 == 0) {
            return false;
        }
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }
}
