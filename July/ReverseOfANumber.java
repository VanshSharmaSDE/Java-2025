import java.util.*;

public class ReverseOfANumber{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number: ");
        int number = sc.nextInt();
        int reversedNumber = 0;
        while(number > 0){
            int lastDigit = number % 10;
            reversedNumber = reversedNumber * 10 + lastDigit;
            number = number / 10;
        }
        System.out.println("Reversed Number: " + reversedNumber);
    }
}