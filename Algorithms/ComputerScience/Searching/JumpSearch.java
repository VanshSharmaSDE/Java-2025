package Algorithms.ComputerScience.Searching;

/**
 * Jump Search Algorithm
 * Time Complexity: O(√n)
 * Space Complexity: O(1)
 * Prerequisites: Array must be sorted
 */
public class JumpSearch {
    
    public static int jumpSearch(int[] arr, int target) {
        int n = arr.length;
        
        // Finding block size to be jumped
        int step = (int) Math.floor(Math.sqrt(n));
        
        // Finding the block where element is present (if it is present)
        int prev = 0;
        while (arr[Math.min(step, n) - 1] < target) {
            prev = step;
            step += (int) Math.floor(Math.sqrt(n));
            if (prev >= n)
                return -1;
        }
        
        // Doing a linear search for target in block beginning with prev
        while (arr[prev] < target) {
            prev++;
            
            // If we reached next block or end of array, element is not present
            if (prev == Math.min(step, n))
                return -1;
        }
        
        // If element is found
        if (arr[prev] == target)
            return prev;
        
        return -1;
    }
    
    public static void main(String[] args) {
        int[] arr = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
        int target = 55;
        
        int index = jumpSearch(arr, target);
        
        if (index == -1) {
            System.out.println("Number " + target + " is not in array");
        } else {
            System.out.println("Number " + target + " is at index " + index);
        }
    }
}
