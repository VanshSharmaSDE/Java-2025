import java.util.Scanner;

public class Question9 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the starting number: ");
        int start = sc.nextInt();
        
        System.out.print("Enter the ending number: ");
        int end = sc.nextInt();
        
        System.out.println("Armstrong numbers between " + start + " and " + end + ":");
        
        boolean found = false;
        for (int i = start; i <= end; i++) {
            if (isArmstrong(i)) {
                System.out.print(i + " ");
                found = true;
            }
        }
        
        if (!found) {
            System.out.println("No Armstrong numbers found in the given range.");
        } else {
            System.out.println();
        }
        
        sc.close();
    }
    
    // Method to check if a number is Armstrong number
    static boolean isArmstrong(int num) {
        int originalNum = num;
        int digits = countDigits(num);
        int sum = 0;
        
        while (num > 0) {
            int digit = num % 10;
            sum += Math.pow(digit, digits);
            num /= 10;
        }
        
        return sum == originalNum;
    }
    
    // Method to count number of digits
    static int countDigits(int num) {
        if (num == 0) return 1;
        
        int count = 0;
        while (num > 0) {
            count++;
            num /= 10;
        }
        return count;
    }
}