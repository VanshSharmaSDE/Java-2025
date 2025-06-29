import java.util.Scanner;

public class ReverseNumber {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        int reversed = reverseNumber(number);
        
        System.out.println("Original number: " + number);
        System.out.println("Reversed number: " + reversed);
        
        scanner.close();
    }
    
    public static int reverseNumber(int n) {
        int reversed = 0;
        boolean isNegative = n < 0;
        n = Math.abs(n);
        
        while (n > 0) {
            reversed = reversed * 10 + n % 10;
            n /= 10;
        }
        
        return isNegative ? -reversed : reversed;
    }
}
