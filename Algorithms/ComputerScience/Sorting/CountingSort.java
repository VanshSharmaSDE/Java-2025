package Algorithms.ComputerScience.Sorting;

/**
 * Counting Sort Algorithm
 * Time Complexity: O(n+k) where k is the range of input
 * Space Complexity: O(k)
 * Stable: Yes
 */
public class CountingSort {
    
    public static void countingSort(int[] arr) {
        int n = arr.length;
        
        // Find the maximum element
        int max = arr[0];
        for (int i = 1; i < n; i++) {
            max = Math.max(max, arr[i]);
        }
        
        // Create count array
        int[] count = new int[max + 1];
        int[] output = new int[n];
        
        // Store count of each element
        for (int i = 0; i < n; i++) {
            count[arr[i]]++;
        }
        
        // Change count[i] so that count[i] now contains actual
        // position of this character in output array
        for (int i = 1; i <= max; i++) {
            count[i] += count[i - 1];
        }
        
        // Build the output array
        for (int i = n - 1; i >= 0; i--) {
            output[count[arr[i]] - 1] = arr[i];
            count[arr[i]]--;
        }
        
        // Copy output array to original array
        System.arraycopy(output, 0, arr, 0, n);
    }
    
    public static void main(String[] args) {
        int[] arr = {4, 2, 2, 8, 3, 3, 1};
        System.out.println("Original array:");
        printArray(arr);
        
        countingSort(arr);
        
        System.out.println("Sorted array:");
        printArray(arr);
    }
    
    public static void printArray(int[] arr) {
        for (int value : arr) {
            System.out.print(value + " ");
        }
        System.out.println();
    }
}
