import java.util.Scanner;

public class CountWords {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a sentence: ");
        String sentence = scanner.nextLine();
        
        int wordCount = countWords(sentence);
        
        System.out.println("Sentence: \"" + sentence + "\"");
        System.out.println("Number of words: " + wordCount);
        
        scanner.close();
    }
    
    public static int countWords(String sentence) {
        // Handle empty or null string
        if (sentence == null || sentence.trim().isEmpty()) {
            return 0;
        }
        
        // Split by one or more whitespace characters
        String[] words = sentence.trim().split("\\s+");
        return words.length;
    }
}
