import java.util.Scanner;

public class ReverseAString {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = sc.nextLine();
        
        // Method 1: Using StringBuilder
        String reversed1 = new StringBuilder(str).reverse().toString();
        
        // Method 2: Using loop
        String reversed2 = "";
        for (int i = str.length() - 1; i >= 0; i--) {
            reversed2 += str.charAt(i);
        }
        
        System.out.println("Original String: " + str);
        System.out.println("Reversed String (StringBuilder): " + reversed1);
        System.out.println("Reversed String (Loop): " + reversed2);
        
        sc.close();
    }
}