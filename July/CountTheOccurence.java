import java.util.*;

public class CountTheOccurence{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int digit = 1233453;
        int number = 3;
        int count = 0;

        while(digit > 0){
            int  lastDigit = digit % 10;
            if(lastDigit == number){
                count++;
            }
            digit = digit / 10;
        }
        System.out.println("The digit " + number + " occurs " + count + " times.");
    }
}