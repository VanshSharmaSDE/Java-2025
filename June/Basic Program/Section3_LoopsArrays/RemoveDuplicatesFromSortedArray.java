import java.util.Scanner;

public class RemoveDuplicatesFromSortedArray {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of sorted array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " sorted elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        System.out.println("Original sorted array: ");
        printArray(array, size);
        
        int newSize = removeDuplicates(array);
        
        System.out.println("Array after removing duplicates: ");
        printArray(array, newSize);
        
        scanner.close();
    }
    
    public static int removeDuplicates(int[] arr) {
        if (arr.length == 0) {
            return 0;
        }
        
        int uniqueIndex = 0;
        
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] != arr[uniqueIndex]) {
                uniqueIndex++;
                arr[uniqueIndex] = arr[i];
            }
        }
        
        return uniqueIndex + 1; // Return new size
    }
    
    public static void printArray(int[] arr, int size) {
        for (int i = 0; i < size; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
