import java.util.Scanner;

public class CheckPalindromeNumber {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        boolean isPalindrome = isPalindromeNumber(number);
        
        if (isPalindrome) {
            System.out.println(number + " is a palindrome number.");
        } else {
            System.out.println(number + " is not a palindrome number.");
        }
        
        scanner.close();
    }
    
    public static boolean isPalindromeNumber(int n) {
        int original = n;
        int reversed = 0;
        
        // Handle negative numbers (not palindromes)
        if (n < 0) {
            return false;
        }
        
        while (n > 0) {
            reversed = reversed * 10 + n % 10;
            n /= 10;
        }
        
        return original == reversed;
    }
}
