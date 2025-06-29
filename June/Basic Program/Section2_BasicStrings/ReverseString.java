import java.util.Scanner;

public class ReverseString {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        String reversed1 = reverseStringMethod1(str);
        String reversed2 = reverseStringMethod2(str);
        
        System.out.println("Original string: " + str);
        System.out.println("Reversed string (Method 1): " + reversed1);
        System.out.println("Reversed string (Method 2): " + reversed2);
        
        scanner.close();
    }
    
    // Method 1: Using StringBuilder
    public static String reverseStringMethod1(String str) {
        return new StringBuilder(str).reverse().toString();
    }
    
    // Method 2: Using character array
    public static String reverseStringMethod2(String str) {
        char[] chars = str.toCharArray();
        int left = 0, right = chars.length - 1;
        
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        
        return new String(chars);
    }
}
