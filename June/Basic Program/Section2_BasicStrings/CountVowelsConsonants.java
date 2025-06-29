import java.util.Scanner;

public class CountVowelsConsonants {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        int[] counts = countVowelsAndConsonants(str);
        
        System.out.println("String: " + str);
        System.out.println("Vowels: " + counts[0]);
        System.out.println("Consonants: " + counts[1]);
        
        scanner.close();
    }
    
    public static int[] countVowelsAndConsonants(String str) {
        int vowels = 0, consonants = 0;
        str = str.toLowerCase();
        
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            
            if (Character.isLetter(ch)) {
                if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') {
                    vowels++;
                } else {
                    consonants++;
                }
            }
        }
        
        return new int[]{vowels, consonants};
    }
}
