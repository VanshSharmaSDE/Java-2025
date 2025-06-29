import java.util.Scanner;

public class RemoveWhiteSpaces {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        String withoutSpaces1 = removeWhiteSpacesMethod1(str);
        String withoutSpaces2 = removeWhiteSpacesMethod2(str);
        
        System.out.println("Original string: \"" + str + "\"");
        System.out.println("Without spaces (Method 1): \"" + withoutSpaces1 + "\"");
        System.out.println("Without spaces (Method 2): \"" + withoutSpaces2 + "\"");
        
        scanner.close();
    }
    
    // Method 1: Using replaceAll()
    public static String removeWhiteSpacesMethod1(String str) {
        return str.replaceAll("\\s+", "");
    }
    
    // Method 2: Using StringBuilder
    public static String removeWhiteSpacesMethod2(String str) {
        StringBuilder result = new StringBuilder();
        
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            if (!Character.isWhitespace(ch)) {
                result.append(ch);
            }
        }
        
        return result.toString();
    }
}
