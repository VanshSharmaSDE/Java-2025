import java.util.Scanner;

public class CalculateDiscountOfProduct {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the original price: ");
        double originalPrice = sc.nextDouble();
        
        System.out.print("Enter the discount percentage: ");
        double discountPercentage = sc.nextDouble();
        
        double discountAmount = (originalPrice * discountPercentage) / 100;
        double finalPrice = originalPrice - discountAmount;
        
        System.out.println("Original Price: Rs. " + originalPrice);
        System.out.println("Discount: " + discountPercentage + "%");
        System.out.println("Discount Amount: Rs. " + String.format("%.2f", discountAmount));
        System.out.println("Final Price: Rs. " + String.format("%.2f", finalPrice));
        
        sc.close();
    }
}