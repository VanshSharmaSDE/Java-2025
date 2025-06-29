import java.util.Scanner;

public class PrintNToOneRecursive {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of N: ");
        int n = scanner.nextInt();
        
        System.out.println("Numbers from " + n + " to 1:");
        printNToOne(n);
        
        scanner.close();
    }
    
    public static void printNToOne(int n) {
        // Base case
        if (n <= 0) {
            return;
        }
        
        // Print current number
        System.out.print(n + " ");
        
        // Recursive call with n-1
        printNToOne(n - 1);
    }
}
