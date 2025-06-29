import java.util.Scanner;

public class ConvertCase {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        String upperCase = convertToUpperCase(str);
        String lowerCase = convertToLowerCase(str);
        String toggleCase = toggleCase(str);
        
        System.out.println("Original string: " + str);
        System.out.println("Uppercase: " + upperCase);
        System.out.println("Lowercase: " + lowerCase);
        System.out.println("Toggle case: " + toggleCase);
        
        scanner.close();
    }
    
    public static String convertToUpperCase(String str) {
        return str.toUpperCase();
    }
    
    public static String convertToLowerCase(String str) {
        return str.toLowerCase();
    }
    
    public static String toggleCase(String str) {
        StringBuilder result = new StringBuilder();
        
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            
            if (Character.isUpperCase(ch)) {
                result.append(Character.toLowerCase(ch));
            } else if (Character.isLowerCase(ch)) {
                result.append(Character.toUpperCase(ch));
            } else {
                result.append(ch);
            }
        }
        
        return result.toString();
    }
}
