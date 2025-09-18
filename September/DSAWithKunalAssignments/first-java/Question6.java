import java.util.Scanner;

public class Question6 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        // Exchange rate (1 USD = 83.5 INR approximately)
        double exchangeRate = 83.5;
        
        System.out.print("Enter amount in Indian Rupees: ");
        double rupees = sc.nextDouble();
        
        double dollars = rupees / exchangeRate;
        
        System.out.println("Amount in USD: $" + String.format("%.2f", dollars));
        System.out.println("Exchange rate used: 1 USD = " + exchangeRate + " INR");
        
        sc.close();
    }
}