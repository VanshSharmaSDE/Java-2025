import java.util.Scanner;

public class PowerOfNumber {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter base (a): ");
        int base = scanner.nextInt();
        System.out.print("Enter exponent (b): ");
        int exponent = scanner.nextInt();
        
        long result1 = powerIterative(base, exponent);
        long result2 = powerRecursive(base, exponent);
        
        System.out.println(base + "^" + exponent + " (iterative) = " + result1);
        System.out.println(base + "^" + exponent + " (recursive) = " + result2);
        System.out.println(base + "^" + exponent + " (built-in) = " + Math.pow(base, exponent));
        
        scanner.close();
    }
    
    // Iterative method
    public static long powerIterative(int base, int exp) {
        if (exp == 0) return 1;
        
        long result = 1;
        for (int i = 0; i < exp; i++) {
            result *= base;
        }
        return result;
    }
    
    // Recursive method
    public static long powerRecursive(int base, int exp) {
        if (exp == 0) return 1;
        if (exp == 1) return base;
        
        return base * powerRecursive(base, exp - 1);
    }
}
