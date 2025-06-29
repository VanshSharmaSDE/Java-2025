import java.util.Scanner;

public class CheckOnlyDigits {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        boolean onlyDigits = containsOnlyDigits(str);
        
        System.out.println("String: \"" + str + "\"");
        
        if (onlyDigits) {
            System.out.println("The string contains only digits.");
        } else {
            System.out.println("The string does not contain only digits.");
        }
        
        scanner.close();
    }
    
    public static boolean containsOnlyDigits(String str) {
        // Handle empty string
        if (str == null || str.isEmpty()) {
            return false;
        }
        
        for (int i = 0; i < str.length(); i++) {
            if (!Character.isDigit(str.charAt(i))) {
                return false;
            }
        }
        
        return true;
    }
}
