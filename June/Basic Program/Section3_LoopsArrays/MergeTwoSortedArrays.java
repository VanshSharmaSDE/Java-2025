import java.util.Scanner;

public class MergeTwoSortedArrays {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter size of first array: ");
        int size1 = scanner.nextInt();
        int[] array1 = new int[size1];
        
        System.out.println("Enter " + size1 + " sorted elements for first array:");
        for (int i = 0; i < size1; i++) {
            array1[i] = scanner.nextInt();
        }
        
        System.out.print("Enter size of second array: ");
        int size2 = scanner.nextInt();
        int[] array2 = new int[size2];
        
        System.out.println("Enter " + size2 + " sorted elements for second array:");
        for (int i = 0; i < size2; i++) {
            array2[i] = scanner.nextInt();
        }
        
        int[] mergedArray = mergeSortedArrays(array1, array2);
        
        System.out.println("First array: ");
        printArray(array1);
        System.out.println("Second array: ");
        printArray(array2);
        System.out.println("Merged sorted array: ");
        printArray(mergedArray);
        
        scanner.close();
    }
    
    public static int[] mergeSortedArrays(int[] arr1, int[] arr2) {
        int[] merged = new int[arr1.length + arr2.length];
        int i = 0, j = 0, k = 0;
        
        // Merge while both arrays have elements
        while (i < arr1.length && j < arr2.length) {
            if (arr1[i] <= arr2[j]) {
                merged[k++] = arr1[i++];
            } else {
                merged[k++] = arr2[j++];
            }
        }
        
        // Copy remaining elements from arr1
        while (i < arr1.length) {
            merged[k++] = arr1[i++];
        }
        
        // Copy remaining elements from arr2
        while (j < arr2.length) {
            merged[k++] = arr2[j++];
        }
        
        return merged;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
