import java.util.Scanner;

public class LCMOfTwoNumbers {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter first number: ");
        int num1 = sc.nextInt();
        
        System.out.print("Enter second number: ");
        int num2 = sc.nextInt();
        
        int lcm = findLCM(num1, num2);
        
        System.out.println("LCM of " + num1 + " and " + num2 + " is: " + lcm);
        
        sc.close();
    }
    
    static int findLCM(int a, int b) {
        return (a * b) / findHCF(a, b);
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