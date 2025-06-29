import java.util.Scanner;

public class BinaryToDecimal {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a binary number: ");
        String binary = scanner.next();
        
        int decimal = binaryToDecimal(binary);
        
        System.out.println("Binary: " + binary);
        System.out.println("Decimal: " + decimal);
        
        // Using built-in method
        System.out.println("Decimal (using built-in): " + Integer.parseInt(binary, 2));
        
        scanner.close();
    }
    
    public static int binaryToDecimal(String binary) {
        int decimal = 0;
        int power = 0;
        
        // Process from right to left
        for (int i = binary.length() - 1; i >= 0; i--) {
            if (binary.charAt(i) == '1') {
                decimal += Math.pow(2, power);
            }
            power++;
        }
        
        return decimal;
    }
}
