import java.math.BigInteger;

public class ArrayToDigit {
    // Converts an array of digits into a number represented as a BigInteger
    public static BigInteger arrayToDigit(int[] digits) {
        StringBuilder sb = new StringBuilder();
        for (int d : digits) {
            if (d < 0 || d > 9) {
                throw new IllegalArgumentException("Invalid digit: " + d);
            }
            sb.append(d);
        }
        return new BigInteger(sb.toString());
    }

    public static void main(String[] args) {
        // Example arrays of any length
        int[] digits1 = {1, 2, 3, 4, 5};
        int[] digits2 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        
        BigInteger number1 = arrayToDigit(digits1);
        BigInteger number2 = arrayToDigit(digits2);
        
        System.out.println("Array 1 converted to digit: " + number1);
        System.out.println("Array 2 converted to digit: " + number2);
    }
}