package Algorithms.ComputerScience.Searching;

/**
 * Binary Search Algorithm (Recursive)
 * Time Complexity: O(log n)
 * Space Complexity: O(log n)
 * Prerequisites: Array must be sorted
 */
public class BinarySearchRecursive {
    
    public static int binarySearch(int[] arr, int left, int right, int target) {
        if (right >= left) {
            int mid = left + (right - left) / 2;
            
            // If the element is present at the middle itself
            if (arr[mid] == target)
                return mid;
            
            // If element is smaller than mid, then it can only be present in left subarray
            if (arr[mid] > target)
                return binarySearch(arr, left, mid - 1, target);
            
            // Else the element can only be present in right subarray
            return binarySearch(arr, mid + 1, right, target);
        }
        
        return -1; // Element not found
    }
    
    public static void main(String[] args) {
        int[] arr = {2, 3, 4, 10, 40};
        int target = 10;
        
        int result = binarySearch(arr, 0, arr.length - 1, target);
        
        if (result == -1) {
            System.out.println("Element not present");
        } else {
            System.out.println("Element found at index " + result);
        }
    }
}
