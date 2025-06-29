import java.util.Scanner;

public class CountSetBits {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = scanner.nextInt();
        
        int setBits = countSetBits(number);
        
        System.out.println("Number: " + number);
        System.out.println("Binary representation: " + Integer.toBinaryString(number));
        System.out.println("Number of set bits (1s): " + setBits);
        
        // Using built-in method
        System.out.println("Set bits (using built-in): " + Integer.bitCount(number));
        
        scanner.close();
    }
    
    public static int countSetBits(int n) {
        int count = 0;
        
        while (n > 0) {
            count += n & 1; // Check if last bit is 1
            n >>= 1; // Right shift by 1
        }
        
        return count;
    }
}
