import java.util.Scanner;

public class HCFOfTwoNumbers {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter first number: ");
        int num1 = sc.nextInt();
        
        System.out.print("Enter second number: ");
        int num2 = sc.nextInt();
        
        int hcf = findHCF(num1, num2);
        
        System.out.println("HCF of " + num1 + " and " + num2 + " is: " + hcf);
        
        sc.close();
    }
    
    static int findHCF(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
}