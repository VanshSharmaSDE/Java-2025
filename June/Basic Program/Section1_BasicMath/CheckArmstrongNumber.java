import java.util.Scanner;

public class CheckArmstrongNumber {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        boolean isArmstrong = isArmstrongNumber(number);
        
        if (isArmstrong) {
            System.out.println(number + " is an Armstrong number.");
        } else {
            System.out.println(number + " is not an Armstrong number.");
        }
        
        scanner.close();
    }
    
    public static boolean isArmstrongNumber(int n) {
        int original = n;
        int numDigits = countDigits(n);
        int sum = 0;
        
        while (n > 0) {
            int digit = n % 10;
            sum += Math.pow(digit, numDigits);
            n /= 10;
        }
        
        return sum == original;
    }
    
    public static int countDigits(int n) {
        if (n == 0) return 1;
        int count = 0;
        while (n > 0) {
            count++;
            n /= 10;
        }
        return count;
    }
}
