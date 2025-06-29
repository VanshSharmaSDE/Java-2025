import java.util.Scanner;

public class DecimalToBinary {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a decimal number: ");
        int decimal = scanner.nextInt();
        
        String binary = decimalToBinary(decimal);
        
        System.out.println("Decimal: " + decimal);
        System.out.println("Binary: " + binary);
        
        // Using built-in method
        System.out.println("Binary (using built-in): " + Integer.toBinaryString(decimal));
        
        scanner.close();
    }
    
    public static String decimalToBinary(int decimal) {
        if (decimal == 0) {
            return "0";
        }
        
        StringBuilder binary = new StringBuilder();
        
        while (decimal > 0) {
            binary.insert(0, decimal % 2);
            decimal /= 2;
        }
        
        return binary.toString();
    }
}
