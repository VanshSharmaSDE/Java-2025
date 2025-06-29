import java.util.LinkedHashSet;
import java.util.Scanner;
import java.util.Set;

public class RemoveDuplicateCharacters {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        String withoutDuplicates = removeDuplicates(str);
        
        System.out.println("Original string: " + str);
        System.out.println("Without duplicates: " + withoutDuplicates);
        
        scanner.close();
    }
    
    public static String removeDuplicates(String str) {
        Set<Character> seen = new LinkedHashSet<>();
        
        for (int i = 0; i < str.length(); i++) {
            seen.add(str.charAt(i));
        }
        
        StringBuilder result = new StringBuilder();
        for (Character ch : seen) {
            result.append(ch);
        }
        
        return result.toString();
    }
}
