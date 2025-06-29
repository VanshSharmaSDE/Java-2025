import java.util.Scanner;

public class Pattern08_CalendarPattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the month (1-12): ");
        int month = scanner.nextInt();
        System.out.print("Enter the year: ");
        int year = scanner.nextInt();
        
        String[] months = {"", "January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"};
        
        int[] daysInMonth = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        
        // Check for leap year
        if (month == 2 && isLeapYear(year)) {
            daysInMonth[2] = 29;
        }
        
        System.out.println("\n" + months[month] + " " + year);
        System.out.println("Su Mo Tu We Th Fr Sa");
        
        // Calculate starting day (simplified - assumes January 1, 2000 was Saturday)
        int startDay = (year - 2000) + (year - 2000) / 4 + month;
        startDay = startDay % 7;
        
        // Print leading spaces
        for (int i = 0; i < startDay; i++) {
            System.out.print("   ");
        }
        
        // Print days
        for (int day = 1; day <= daysInMonth[month]; day++) {
            System.out.printf("%2d ", day);
            if ((day + startDay) % 7 == 0) {
                System.out.println();
            }
        }
        System.out.println();
        
        scanner.close();
    }
    
    public static boolean isLeapYear(int year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }
}
