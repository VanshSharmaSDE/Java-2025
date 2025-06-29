import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class CharacterFrequency {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        Map<Character, Integer> frequency = getCharacterFrequency(str);
        
        System.out.println("Character frequency in \"" + str + "\":");
        for (Map.Entry<Character, Integer> entry : frequency.entrySet()) {
            System.out.println("'" + entry.getKey() + "' : " + entry.getValue());
        }
        
        scanner.close();
    }
    
    public static Map<Character, Integer> getCharacterFrequency(String str) {
        Map<Character, Integer> frequency = new HashMap<>();
        
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            frequency.put(ch, frequency.getOrDefault(ch, 0) + 1);
        }
        
        return frequency;
    }
}
