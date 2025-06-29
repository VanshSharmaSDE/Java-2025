import java.util.Scanner;

public class GCDandLCM {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter first number: ");
        int num1 = scanner.nextInt();
        System.out.print("Enter second number: ");
        int num2 = scanner.nextInt();
        
        int gcd = findGCD(num1, num2);
        int lcm = findLCM(num1, num2, gcd);
        
        System.out.println("GCD of " + num1 + " and " + num2 + " is: " + gcd);
        System.out.println("LCM of " + num1 + " and " + num2 + " is: " + lcm);
        
        scanner.close();
    }
    
    // Euclidean algorithm for GCD
    public static int findGCD(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    
    // LCM using the formula: LCM(a,b) = (a*b)/GCD(a,b)
    public static int findLCM(int a, int b, int gcd) {
        return (a * b) / gcd;
    }
}
