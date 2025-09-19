import java.util.Scanner;

public class CheckLeapYear {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter a year: ");
        int year = sc.nextInt();
        
        if (isLeapYear(year)) {
            System.out.println(year + " is a leap year.");
        } else {
            System.out.println(year + " is not a leap year.");
        }
        
        sc.close();
    }
    
    static boolean isLeapYear(int year) {
        // A year is leap if:
        // 1. It is divisible by 4 AND not divisible by 100
        // 2. OR it is divisible by 400
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }
}