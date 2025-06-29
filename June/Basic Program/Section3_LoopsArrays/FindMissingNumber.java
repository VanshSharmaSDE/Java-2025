import java.util.Scanner;

public class FindMissingNumber {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of N: ");
        int n = scanner.nextInt();
        
        int[] array = new int[n - 1]; // Array with one missing number
        
        System.out.println("Enter " + (n - 1) + " elements (numbers from 1 to " + n + " with one missing):");
        for (int i = 0; i < n - 1; i++) {
            array[i] = scanner.nextInt();
        }
        
        int missingNumber = findMissingNumber(array, n);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Missing number: " + missingNumber);
        
        scanner.close();
    }
    
    public static int findMissingNumber(int[] arr, int n) {
        // Method 1: Using sum formula
        int expectedSum = n * (n + 1) / 2;
        int actualSum = 0;
        
        for (int i = 0; i < arr.length; i++) {
            actualSum += arr[i];
        }
        
        return expectedSum - actualSum;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
