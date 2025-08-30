package Algorithms.DataStructures.Heaps;

/**
 * Binary Heap Algorithms and Advanced Heap Operations
 */
public class HeapAlgorithms {
    
    /**
     * Heap Sort implementation
     * Time Complexity: O(n log n)
     * Space Complexity: O(1)
     */
    public static void heapSort(int[] arr) {
        int n = arr.length;
        
        // Build max heap
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // Extract elements one by one
        for (int i = n - 1; i > 0; i--) {
            // Move current root to end
            swap(arr, 0, i);
            
            // Call heapify on reduced heap
            heapify(arr, i, 0);
        }
    }
    
    /**
     * Heapify a subtree rooted at index i
     * Time Complexity: O(log n)
     */
    private static void heapify(int[] arr, int n, int i) {
        int largest = i; // Initialize largest as root
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        // If left child exists and is greater than root
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        // If right child exists and is greater than largest so far
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        // If largest is not root
        if (largest != i) {
            swap(arr, i, largest);
            
            // Recursively heapify the affected sub-tree
            heapify(arr, n, largest);
        }
    }
    
    /**
     * Find k largest elements in array using min heap
     * Time Complexity: O(n log k)
     * Space Complexity: O(k)
     */
    public static int[] findKLargest(int[] arr, int k) {
        if (k > arr.length) {
            throw new IllegalArgumentException("k is greater than array length");
        }
        
        // Create a min heap of size k
        int[] heap = new int[k];
        int heapSize = 0;
        
        // Insert first k elements
        for (int i = 0; i < k; i++) {
            heap[heapSize++] = arr[i];
        }
        
        // Build min heap
        for (int i = k / 2 - 1; i >= 0; i--) {
            minHeapify(heap, k, i);
        }
        
        // Process remaining elements
        for (int i = k; i < arr.length; i++) {
            if (arr[i] > heap[0]) {
                heap[0] = arr[i];
                minHeapify(heap, k, 0);
            }
        }
        
        return heap;
    }
    
    /**
     * Find k smallest elements in array using max heap
     * Time Complexity: O(n log k)
     * Space Complexity: O(k)
     */
    public static int[] findKSmallest(int[] arr, int k) {
        if (k > arr.length) {
            throw new IllegalArgumentException("k is greater than array length");
        }
        
        // Create a max heap of size k
        int[] heap = new int[k];
        int heapSize = 0;
        
        // Insert first k elements
        for (int i = 0; i < k; i++) {
            heap[heapSize++] = arr[i];
        }
        
        // Build max heap
        for (int i = k / 2 - 1; i >= 0; i--) {
            maxHeapify(heap, k, i);
        }
        
        // Process remaining elements
        for (int i = k; i < arr.length; i++) {
            if (arr[i] < heap[0]) {
                heap[0] = arr[i];
                maxHeapify(heap, k, 0);
            }
        }
        
        return heap;
    }
    
    /**
     * Check if array represents a valid min heap
     * Time Complexity: O(n)
     */
    public static boolean isMinHeap(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i <= (n - 2) / 2; i++) {
            // Left child
            if (2 * i + 1 < n && arr[i] > arr[2 * i + 1]) {
                return false;
            }
            
            // Right child
            if (2 * i + 2 < n && arr[i] > arr[2 * i + 2]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Check if array represents a valid max heap
     * Time Complexity: O(n)
     */
    public static boolean isMaxHeap(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i <= (n - 2) / 2; i++) {
            // Left child
            if (2 * i + 1 < n && arr[i] < arr[2 * i + 1]) {
                return false;
            }
            
            // Right child
            if (2 * i + 2 < n && arr[i] < arr[2 * i + 2]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Merge k sorted arrays using min heap
     * Time Complexity: O(n log k) where n is total elements
     * Space Complexity: O(k)
     */
    public static int[] mergeKSortedArrays(int[][] arrays) {
        // Create min heap of HeapNode
        java.util.PriorityQueue<HeapNode> heap = new java.util.PriorityQueue<>(
            (a, b) -> a.value - b.value
        );
        
        int totalElements = 0;
        
        // Add first element of each array to heap
        for (int i = 0; i < arrays.length; i++) {
            if (arrays[i].length > 0) {
                heap.offer(new HeapNode(arrays[i][0], i, 0));
                totalElements += arrays[i].length;
            }
        }
        
        int[] result = new int[totalElements];
        int index = 0;
        
        while (!heap.isEmpty()) {
            HeapNode node = heap.poll();
            result[index++] = node.value;
            
            // Add next element from same array
            if (node.elementIndex + 1 < arrays[node.arrayIndex].length) {
                heap.offer(new HeapNode(
                    arrays[node.arrayIndex][node.elementIndex + 1],
                    node.arrayIndex,
                    node.elementIndex + 1
                ));
            }
        }
        
        return result;
    }
    
    /**
     * Convert min heap to max heap
     * Time Complexity: O(n)
     */
    public static void convertMinToMaxHeap(int[] arr) {
        // Start from last non-leaf node and heapify
        for (int i = (arr.length - 2) / 2; i >= 0; i--) {
            maxHeapify(arr, arr.length, i);
        }
    }
    
    /**
     * Convert max heap to min heap
     * Time Complexity: O(n)
     */
    public static void convertMaxToMinHeap(int[] arr) {
        // Start from last non-leaf node and heapify
        for (int i = (arr.length - 2) / 2; i >= 0; i--) {
            minHeapify(arr, arr.length, i);
        }
    }
    
    /**
     * Find median in a stream of integers using two heaps
     */
    public static class MedianFinder {
        private java.util.PriorityQueue<Integer> maxHeap; // Lower half
        private java.util.PriorityQueue<Integer> minHeap; // Upper half
        
        public MedianFinder() {
            maxHeap = new java.util.PriorityQueue<>((a, b) -> b - a); // Max heap
            minHeap = new java.util.PriorityQueue<>(); // Min heap
        }
        
        public void addNumber(int num) {
            if (maxHeap.isEmpty() || num <= maxHeap.peek()) {
                maxHeap.offer(num);
            } else {
                minHeap.offer(num);
            }
            
            // Balance heaps
            if (maxHeap.size() > minHeap.size() + 1) {
                minHeap.offer(maxHeap.poll());
            } else if (minHeap.size() > maxHeap.size() + 1) {
                maxHeap.offer(minHeap.poll());
            }
        }
        
        public double findMedian() {
            if (maxHeap.size() == minHeap.size()) {
                return (maxHeap.peek() + minHeap.peek()) / 2.0;
            } else if (maxHeap.size() > minHeap.size()) {
                return maxHeap.peek();
            } else {
                return minHeap.peek();
            }
        }
    }
    
    // Helper classes and methods
    private static class HeapNode {
        int value;
        int arrayIndex;
        int elementIndex;
        
        HeapNode(int value, int arrayIndex, int elementIndex) {
            this.value = value;
            this.arrayIndex = arrayIndex;
            this.elementIndex = elementIndex;
        }
    }
    
    private static void minHeapify(int[] arr, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] < arr[smallest]) {
            smallest = left;
        }
        
        if (right < n && arr[right] < arr[smallest]) {
            smallest = right;
        }
        
        if (smallest != i) {
            swap(arr, i, smallest);
            minHeapify(arr, n, smallest);
        }
    }
    
    private static void maxHeapify(int[] arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest != i) {
            swap(arr, i, largest);
            maxHeapify(arr, n, largest);
        }
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    private static void printArray(int[] arr) {
        for (int value : arr) {
            System.out.print(value + " ");
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Heap Algorithms Demo:");
        System.out.println("=====================");
        
        // Heap Sort
        System.out.println("1. Heap Sort:");
        int[] arr = {12, 11, 13, 5, 6, 7};
        System.out.print("Original array: ");
        printArray(arr);
        
        heapSort(arr);
        System.out.print("Sorted array: ");
        printArray(arr);
        
        // Find K Largest
        System.out.println("\n2. Find K Largest:");
        int[] arr2 = {7, 10, 4, 3, 20, 15};
        System.out.print("Array: ");
        printArray(arr2);
        
        int[] kLargest = findKLargest(arr2, 3);
        System.out.print("3 largest elements: ");
        printArray(kLargest);
        
        // Find K Smallest
        System.out.println("\n3. Find K Smallest:");
        int[] kSmallest = findKSmallest(arr2, 3);
        System.out.print("3 smallest elements: ");
        printArray(kSmallest);
        
        // Check heap property
        System.out.println("\n4. Check Heap Property:");
        int[] minHeapArr = {1, 3, 6, 5, 2, 4};
        int[] maxHeapArr = {10, 8, 9, 4, 7, 6};
        
        System.out.print("Array: ");
        printArray(minHeapArr);
        System.out.println("Is min heap: " + isMinHeap(minHeapArr));
        
        System.out.print("Array: ");
        printArray(maxHeapArr);
        System.out.println("Is max heap: " + isMaxHeap(maxHeapArr));
        
        // Merge k sorted arrays
        System.out.println("\n5. Merge K Sorted Arrays:");
        int[][] arrays = {
            {1, 4, 7},
            {2, 5, 8},
            {3, 6, 9}
        };
        
        System.out.println("Input arrays:");
        for (int[] array : arrays) {
            printArray(array);
        }
        
        int[] merged = mergeKSortedArrays(arrays);
        System.out.print("Merged array: ");
        printArray(merged);
        
        // Convert min heap to max heap
        System.out.println("\n6. Convert Min Heap to Max Heap:");
        int[] minHeap = {1, 3, 6, 5, 2, 4};
        System.out.print("Min heap: ");
        printArray(minHeap);
        
        convertMinToMaxHeap(minHeap);
        System.out.print("Max heap: ");
        printArray(minHeap);
        
        // Median finder
        System.out.println("\n7. Median Finder:");
        MedianFinder medianFinder = new MedianFinder();
        int[] stream = {5, 15, 1, 3, 9, 8, 7, 2, 10};
        
        for (int num : stream) {
            medianFinder.addNumber(num);
            System.out.println("Added " + num + ", median: " + medianFinder.findMedian());
        }
    }
}
