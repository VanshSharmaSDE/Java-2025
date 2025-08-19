import java.util.*;

public class SearchinStrings{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String input = sc.nextLine();

        System.out.print("Enter a character to search for: ");
        char searchChar = sc.next().charAt(0);

        // Search for the character in the string
        int index = input.indexOf(searchChar);
        if (index != -1) {
            System.out.println("Character '" + searchChar + "' found at index: " + index);
        } else {
            System.out.println("Character '" + searchChar + "' not found.");
        }

        // Without using indexOf method
        char[] charArray = input.toCharArray();

        int foundIndex = -1;
        for (int i = 0; i < charArray.length; i++) {
            if(searchChar == charArray[i]){
                foundIndex = i;
                break;
            }
        }

        if (foundIndex != -1) {
            System.out.println("Character '" + searchChar + "' found at index: " + foundIndex);
        } else {
            System.out.println("Character '" + searchChar + "' not found.");
        }

        sc.close();
    }
}