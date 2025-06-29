import java.util.Arrays;
import java.util.Scanner;

public class CheckAnagrams {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter first string: ");
        String str1 = scanner.nextLine();
        System.out.print("Enter second string: ");
        String str2 = scanner.nextLine();
        
        boolean areAnagrams = checkAnagrams(str1, str2);
        
        System.out.println("String 1: " + str1);
        System.out.println("String 2: " + str2);
        
        if (areAnagrams) {
            System.out.println("The strings are anagrams.");
        } else {
            System.out.println("The strings are not anagrams.");
        }
        
        scanner.close();
    }
    
    public static boolean checkAnagrams(String str1, String str2) {
        // Remove spaces and convert to lowercase
        str1 = str1.replaceAll("\\s+", "").toLowerCase();
        str2 = str2.replaceAll("\\s+", "").toLowerCase();
        
        // If lengths are different, they can't be anagrams
        if (str1.length() != str2.length()) {
            return false;
        }
        
        // Convert to character arrays and sort
        char[] chars1 = str1.toCharArray();
        char[] chars2 = str2.toCharArray();
        
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        
        // Compare sorted arrays
        return Arrays.equals(chars1, chars2);
    }
}
