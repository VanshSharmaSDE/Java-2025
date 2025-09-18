/*
 * Program to find HCF (Highest Common Factor) and LCM (Least Common Multiple) of two numbers
 * 
 * HCF (also called GCD - Greatest Common Divisor): The largest positive integer that 
 * divides both numbers without leaving a remainder.
 * 
 * LCM: The smallest positive integer that is divisible by both numbers.
 * 
 * Mathematical relationship: LCM(a,b) = (a * b) / HCF(a,b)
 */

public class Question4{
    public static void main(String[] args) {
        // Initialize two sample numbers for demonstration
        int num1 = 12, num2 = 18; // Example numbers
        
        // Calculate HCF using the Euclidean algorithm
        int hcf = findHCF(num1, num2);
        
        // Calculate LCM using the mathematical formula: LCM = (num1 * num2) / HCF
        int lcm = (num1 * num2) / hcf; // Formula: LCM = (num1 * num2) / HCF

        // Display the results
        System.out.println("HCF of " + num1 + " and " + num2 + " is: " + hcf);
        System.out.println("LCM of " + num1 + " and " + num2 + " is: " + lcm);
    }

    /**
     * Method to calculate HCF using the Euclidean algorithm
     * 
     * The Euclidean algorithm works on the principle that:
     * HCF(a, b) = HCF(b, a % b) until b becomes 0
     * 
     * Example: HCF(12, 18)
     * Step 1: HCF(12, 18) -> a=12, b=18
     * Step 2: a=18, b=12%18=12 -> HCF(18, 12)
     * Step 3: a=12, b=18%12=6 -> HCF(12, 6)
     * Step 4: a=6, b=12%6=0 -> HCF(6, 0)
     * Step 5: b=0, so return a=6
     * 
     * @param a First number
     * @param b Second number
     * @return HCF of the two numbers
     */
    public static int findHCF(int a, int b) {
        // Continue the algorithm until b becomes 0
        while (b != 0) {
            // Store the current value of b in a temporary variable
            int temp = b;
            
            // Update b to be the remainder of a divided by b
            b = a % b;
            
            // Update a to be the previous value of b
            a = temp;
        }
        
        // When b becomes 0, a contains the HCF
        return a;
    }
}