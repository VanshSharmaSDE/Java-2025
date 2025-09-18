public class Question4{
    public static void main(String[] args) {
        int num1 = 12, num2 = 18; // Example numbers
        int hcf = findHCF(num1, num2);
        int lcm = (num1 * num2) / hcf; // Formula: LCM = (num1 * num2) / HCF

        System.out.println("HCF of " + num1 + " and " + num2 + " is: " + hcf);
        System.out.println("LCM of " + num1 + " and " + num2 + " is: " + lcm);
    }

    // Method to calculate HCF using the Euclidean algorithm
    public static int findHCF(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;

    }
}