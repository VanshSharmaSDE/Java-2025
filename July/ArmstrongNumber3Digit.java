import java.util.*;

public class ArmstrongNumber3Digit{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the Number: ");
        int number = sc.nextInt();
        int originalNumber = number;
        int testNumber = 0;
        int digit = 0;
        while(number > 0){
           digit = number%10;
           int cube = digit * digit * digit;
           testNumber += cube;  
           number = number/10;
        }
        if(testNumber == originalNumber){
            System.out.println("The number is an Armstrong number.");
        }else{
            System.out.println("The number is not an Armstrong number.");
        }
    }
}