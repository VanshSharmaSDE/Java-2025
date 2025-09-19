import java.util.Scanner;

public class KunalDaysOut {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the month (1-12): ");
        int month = sc.nextInt();
        
        System.out.print("Enter the year: ");
        int year = sc.nextInt();
        
        int daysInMonth = getDaysInMonth(month, year);
        int evenDays = 0;
        
        if (daysInMonth == -1) {
            System.out.println("Invalid month entered!");
            return;
        }
        
        System.out.println("Days Kunal can go out in month " + month + " of year " + year + ":");
        
        for (int day = 1; day <= daysInMonth; day++) {
            if (day % 2 == 0) {
                System.out.print(day + " ");
                evenDays++;
            }
        }
        
        System.out.println("\nTotal number of days Kunal can go out: " + evenDays);
        
        sc.close();
    }
    
    static int getDaysInMonth(int month, int year) {
        switch (month) {
            case 1: case 3: case 5: case 7: case 8: case 10: case 12:
                return 31;
            case 4: case 6: case 9: case 11:
                return 30;
            case 2:
                return isLeapYear(year) ? 29 : 28;
            default:
                return -1; // Invalid month
        }
    }
    
    static boolean isLeapYear(int year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }
}