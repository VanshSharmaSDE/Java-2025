import java.util.Scanner;

public class CompoundInterest {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter principal amount: ");
        double principal = sc.nextDouble();
        
        System.out.print("Enter rate of interest: ");
        double rate = sc.nextDouble();
        
        System.out.print("Enter time (in years): ");
        double time = sc.nextDouble();
        
        System.out.print("Enter number of times interest compounds per year: ");
        int n = sc.nextInt();
        
        // A = P(1 + r/n)^(nt)
        double amount = principal * Math.pow((1 + rate / (100 * n)), n * time);
        double compoundInterest = amount - principal;
        
        System.out.println("Principal: Rs. " + principal);
        System.out.println("Rate: " + rate + "%");
        System.out.println("Time: " + time + " years");
        System.out.println("Compound Interest: Rs. " + String.format("%.2f", compoundInterest));
        System.out.println("Final Amount: Rs. " + String.format("%.2f", amount));
        
        sc.close();
    }
}