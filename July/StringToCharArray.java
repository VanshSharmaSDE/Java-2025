import java.util.*;

public class StringToCharArray {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String input = sc.nextLine();

        // By  using Pre Defined Method
        // Convert string to character array
        char[] charArray = input.toCharArray();

        // Print the character array
        System.out.println("Character Array: " + Arrays.toString(charArray));

        // Example of accessing characters in the array
        System.out.println("First character: " + charArray[0]);
        System.out.println("Length of character array: " + charArray.length);

        // By using Manual Method
        char[] manualArray = new char[input.length()];
        for (int i = 0; i < input.length(); i++) {
            manualArray[i] = input.charAt(i);
        }
        System.out.println("Character Array (Manual): " + Arrays.toString(manualArray));

        sc.close();
    }
}