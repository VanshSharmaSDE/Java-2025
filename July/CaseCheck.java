import java.util.*;

public class CaseCheck{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the Character: ");

        // By TypeCasting the input to char
        // char ch = sc.next().charAt(0); // dont remove space
        char ch = sc.next().trim().charAt(0); // to avoid leading/trailing spaces
        int asciiValue = (int)(ch);
        if (asciiValue >=65 && asciiValue <=90) {
            System.out.println("Uppercase Character");
        } else if (asciiValue >= 97 && asciiValue <= 122) {
            System.out.println("Lowercase Character");
        } else if (asciiValue >= 48 && asciiValue <= 57) {
            System.out.println("Digit");
        } else if ((asciiValue >= 32 && asciiValue <= 47) || (asciiValue >= 58 && asciiValue <= 64) || 
                   (asciiValue >= 91 && asciiValue <= 96) || (asciiValue >= 123 && asciiValue <= 126)) {
            System.out.println("Special Character");
            
        } else {
            System.out.println("Unknown Character");
        }

        // using Character.isUpperCase() and Character.isLowerCase()
        if (Character.isUpperCase(ch)) {
            System.out.println("Uppercase Character using Character.isUpperCase()");
        } else if (Character.isLowerCase(ch)) {
            System.out.println("Lowercase Character using Character.isLowerCase()");
        } else if (Character.isDigit(ch)) {
            System.out.println("Digit using Character.isDigit()");
        } else {
            System.out.println("Special Character using Character methods");
        }
    }
}