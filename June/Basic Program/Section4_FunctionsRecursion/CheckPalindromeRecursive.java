import java.util.Scanner;

public class CheckPalindromeRecursive {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("Choose an option:");
        System.out.println("1. Check if a number is palindrome");
        System.out.println("2. Check if a string is palindrome");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        
        switch (choice) {
            case 1:
                System.out.print("Enter a number: ");
                int number = scanner.nextInt();
                boolean isNumberPalindrome = isPalindromeNumber(number);
                if (isNumberPalindrome) {
                    System.out.println(number + " is a palindrome number.");
                } else {
                    System.out.println(number + " is not a palindrome number.");
                }
                break;
                
            case 2:
                scanner.nextLine(); // Consume newline
                System.out.print("Enter a string: ");
                String str = scanner.nextLine();
                boolean isStringPalindrome = isPalindromeString(str.toLowerCase(), 0, str.length() - 1);
                if (isStringPalindrome) {
                    System.out.println("\"" + str + "\" is a palindrome string.");
                } else {
                    System.out.println("\"" + str + "\" is not a palindrome string.");
                }
                break;
                
            default:
                System.out.println("Invalid choice!");
        }
        
        scanner.close();
    }
    
    // Check if a number is palindrome using recursion
    public static boolean isPalindromeNumber(int n) {
        if (n < 0) {
            return false; // Negative numbers are not palindromes
        }
        
        int reversed = reverseNumber(n, 0);
        return n == reversed;
    }
    
    // Helper method to reverse a number recursively
    public static int reverseNumber(int n, int reversed) {
        // Base case
        if (n == 0) {
            return reversed;
        }
        
        // Recursive case
        return reverseNumber(n / 10, reversed * 10 + n % 10);
    }
    
    // Check if a string is palindrome using recursion
    public static boolean isPalindromeString(String str, int left, int right) {
        // Base case: if left crosses right, it's a palindrome
        if (left >= right) {
            return true;
        }
        
        // If characters don't match, it's not a palindrome
        if (str.charAt(left) != str.charAt(right)) {
            return false;
        }
        
        // Recursive case: check the substring
        return isPalindromeString(str, left + 1, right - 1);
    }
}
