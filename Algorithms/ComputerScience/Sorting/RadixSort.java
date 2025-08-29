package Algorithms.ComputerScience.Sorting;

/**
 * Radix Sort Algorithm
 * Time Complexity: O(d*(n+b)) where d is number of digits, b is base
 * Space Complexity: O(n+b)
 * Stable: Yes
 */
public class RadixSort {
    
    public static void radixSort(int[] arr) {
        // Find the maximum number to know number of digits
        int max = getMax(arr);
        
        // Do counting sort for every digit
        for (int exp = 1; max / exp > 0; exp *= 10)
            countSort(arr, exp);
    }
    
    private static int getMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length; i++)
            if (arr[i] > max)
                max = arr[i];
        return max;
    }
    
    private static void countSort(int[] arr, int exp) {
        int n = arr.length;
        int[] output = new int[n];
        int[] count = new int[10];
        
        // Store count of occurrences in count[]
        for (int i = 0; i < n; i++)
            count[(arr[i] / exp) % 10]++;
        
        // Change count[i] so that count[i] now contains actual
        // position of this digit in output[]
        for (int i = 1; i < 10; i++)
            count[i] += count[i - 1];
        
        // Build the output array
        for (int i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }
        
        // Copy the output array to arr[]
        System.arraycopy(output, 0, arr, 0, n);
    }
    
    public static void main(String[] args) {
        int[] arr = {170, 45, 75, 90, 2, 802, 24, 66};
        System.out.println("Original array:");
        printArray(arr);
        
        radixSort(arr);
        
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
