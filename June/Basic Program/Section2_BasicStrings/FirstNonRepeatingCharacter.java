import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class FirstNonRepeatingCharacter {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        char firstNonRepeating = findFirstNonRepeatingCharacter(str);
        
        System.out.println("String: " + str);
        if (firstNonRepeating != '\0') {
            System.out.println("First non-repeating character: '" + firstNonRepeating + "'");
        } else {
            System.out.println("No non-repeating character found.");
        }
        
        scanner.close();
    }
    
    public static char findFirstNonRepeatingCharacter(String str) {
        Map<Character, Integer> frequency = new HashMap<>();
        
        // Count frequency of each character
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            frequency.put(ch, frequency.getOrDefault(ch, 0) + 1);
        }
        
        // Find first character with frequency 1
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            if (frequency.get(ch) == 1) {
                return ch;
            }
        }
        
        return '\0'; // No non-repeating character found
    }
}
