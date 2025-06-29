import java.util.Scanner;

public class MaxProductOfTwoElements {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        int maxProduct = findMaxProduct(array);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Maximum product of two elements: " + maxProduct);
        
        scanner.close();
    }
    
    public static int findMaxProduct(int[] arr) {
        if (arr.length < 2) {
            throw new IllegalArgumentException("Array must have at least 2 elements");
        }
        
        int maxProduct = Integer.MIN_VALUE;
        
        // Check all pairs
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                int product = arr[i] * arr[j];
                if (product > maxProduct) {
                    maxProduct = product;
                }
            }
        }
        
        return maxProduct;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
