import java.util.Scanner;

public class CalculateBattingAverage {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter total runs scored: ");
        int totalRuns = sc.nextInt();
        
        System.out.print("Enter number of times out: ");
        int timesOut = sc.nextInt();
        
        if (timesOut == 0) {
            System.out.println("Batting Average: Not Out (Infinite)");
        } else {
            double battingAverage = (double) totalRuns / timesOut;
            System.out.println("Batting Average: " + String.format("%.2f", battingAverage));
        }
        
        sc.close();
    }
}