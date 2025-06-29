import java.util.Scanner;

public class CountDigits {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        int count = countDigits(number);
        
        System.out.println("Number of digits in " + number + " is: " + count);
        
        scanner.close();
    }
    
    public static int countDigits(int n) {
        // Handle negative numbers
        n = Math.abs(n);
        
        // Special case for 0
        if (n == 0) {
            return 1;
        }
        
        int count = 0;
        while (n > 0) {
            count++;
            n /= 10;
        }
        return count;
    }
}
