import java.util.Scanner;

public class CalculateDepreciationOfValue {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the initial value: ");
        double initialValue = sc.nextDouble();
        
        System.out.print("Enter the depreciation rate (percentage): ");
        double depreciationRate = sc.nextDouble();
        
        System.out.print("Enter the number of years: ");
        int years = sc.nextInt();
        
        double currentValue = initialValue;
        
        System.out.println("\nDepreciation Schedule:");
        System.out.println("Year 0: Rs. " + String.format("%.2f", currentValue));
        
        for (int i = 1; i <= years; i++) {
            double depreciation = (currentValue * depreciationRate) / 100;
            currentValue -= depreciation;
            System.out.println("Year " + i + ": Rs. " + String.format("%.2f", currentValue));
        }
        
        double totalDepreciation = initialValue - currentValue;
        System.out.println("\nTotal Depreciation: Rs. " + String.format("%.2f", totalDepreciation));
        
        sc.close();
    }
}