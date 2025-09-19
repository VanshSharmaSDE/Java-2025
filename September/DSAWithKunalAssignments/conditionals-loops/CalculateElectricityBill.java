import java.util.Scanner;

public class CalculateElectricityBill {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter units consumed: ");
        double units = sc.nextDouble();
        
        double bill = 0;
        
        // Rate structure
        if (units <= 50) {
            bill = units * 0.50;
        } else if (units <= 150) {
            bill = 50 * 0.50 + (units - 50) * 0.75;
        } else if (units <= 250) {
            bill = 50 * 0.50 + 100 * 0.75 + (units - 150) * 1.20;
        } else {
            bill = 50 * 0.50 + 100 * 0.75 + 100 * 1.20 + (units - 250) * 1.50;
        }
        
        // Add 20% surcharge if bill > 400
        if (bill > 400) {
            bill += bill * 0.20;
        }
        
        System.out.println("Electricity Bill: Rs. " + String.format("%.2f", bill));
        
        sc.close();
    }
}