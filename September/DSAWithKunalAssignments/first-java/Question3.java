import java.util.Scanner;

public class Question3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter Principal amount: ");
        double principal = sc.nextDouble();
        
        System.out.print("Enter Time (in years): ");
        double time = sc.nextDouble();
        
        System.out.print("Enter Rate of interest: ");
        double rate = sc.nextDouble();
        
        double simpleInterest = (principal * time * rate) / 100;
        
        System.out.println("Simple Interest = " + simpleInterest);
        System.out.println("Total Amount = " + (principal + simpleInterest));
        
        sc.close();
    }
}