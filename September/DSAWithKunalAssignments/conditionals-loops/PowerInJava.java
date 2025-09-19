import java.util.Scanner;

public class PowerInJava {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base: ");
        double base = sc.nextDouble();
        
        System.out.print("Enter the exponent: ");
        double exponent = sc.nextDouble();
        
        // Using Math.pow()
        double result1 = Math.pow(base, exponent);
        
        // Using manual calculation for integer exponents
        double result2 = 1;
        if (exponent >= 0 && exponent == (int)exponent) {
            for (int i = 1; i <= exponent; i++) {
                result2 *= base;
            }
            System.out.println(base + " raised to " + exponent + " (manual calculation) = " + result2);
        }
        
        System.out.println(base + " raised to " + exponent + " (Math.pow) = " + result1);
        
        sc.close();
    }
}