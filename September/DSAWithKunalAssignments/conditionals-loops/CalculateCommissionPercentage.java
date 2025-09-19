import java.util.Scanner;

public class CalculateCommissionPercentage {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter total sales amount: ");
        double totalSales = sc.nextDouble();
        
        System.out.print("Enter commission percentage: ");
        double commissionRate = sc.nextDouble();
        
        double commission = (totalSales * commissionRate) / 100;
        
        System.out.println("Total Sales: Rs. " + totalSales);
        System.out.println("Commission Rate: " + commissionRate + "%");
        System.out.println("Commission Amount: Rs. " + String.format("%.2f", commission));
        
        sc.close();
    }
}