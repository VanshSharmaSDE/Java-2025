import java.util.Scanner;

public class CalculateAverageOfNNumbers {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the number of values: ");
        int n = sc.nextInt();
        
        if (n <= 0) {
            System.out.println("Please enter a positive number.");
            return;
        }
        
        double sum = 0;
        
        for (int i = 1; i <= n; i++) {
            System.out.print("Enter number " + i + ": ");
            double number = sc.nextDouble();
            sum += number;
        }
        
        double average = sum / n;
        
        System.out.println("Sum: " + sum);
        System.out.println("Average: " + String.format("%.2f", average));
        
        sc.close();
    }
}