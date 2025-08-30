package Algorithms.DataStructures.Heaps;

/**
 * Max Heap Implementation with heap algorithms
 */
public class MaxHeap<T extends Comparable<T>> {
    
    private T[] heap;
    private int size;
    private int capacity;
    
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public MaxHeap() {
        this.capacity = DEFAULT_CAPACITY;
        this.heap = (T[]) new Comparable[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public MaxHeap(int capacity) {
        this.capacity = capacity;
        this.heap = (T[]) new Comparable[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public MaxHeap(T[] array) {
        this.capacity = array.length;
        this.size = array.length;
        this.heap = (T[]) new Comparable[capacity];
        System.arraycopy(array, 0, heap, 0, size);
        buildHeap();
    }
    
    /**
     * Insert element into max heap
     * Time Complexity: O(log n)
     */
    public void insert(T item) {
        if (size >= capacity) {
            resize();
        }
        
        heap[size] = item;
        heapifyUp(size);
        size++;
    }
    
    /**
     * Extract maximum element (root)
     * Time Complexity: O(log n)
     */
    public T extractMax() {
        if (isEmpty()) {
            throw new RuntimeException("Heap is empty");
        }
        
        T max = heap[0];
        heap[0] = heap[size - 1];
        heap[size - 1] = null;
        size--;
        
        if (size > 0) {
            heapifyDown(0);
        }
        
        return max;
    }
    
    /**
     * Peek at maximum element without removing
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new RuntimeException("Heap is empty");
        }
        return heap[0];
    }
    
    /**
     * Delete element at specific index
     * Time Complexity: O(log n)
     */
    public void delete(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == size - 1) {
            heap[index] = null;
            size--;
            return;
        }
        
        heap[index] = heap[size - 1];
        heap[size - 1] = null;
        size--;
        
        heapifyUp(index);
        heapifyDown(index);
    }
    
    /**
     * Increase key at specific index
     * Time Complexity: O(log n)
     */
    public void increaseKey(int index, T newValue) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (newValue.compareTo(heap[index]) < 0) {
            throw new IllegalArgumentException("New value is smaller than current value");
        }
        
        heap[index] = newValue;
        heapifyUp(index);
    }
    
    /**
     * Build heap from array in-place
     * Time Complexity: O(n)
     */
    public void buildHeap() {
        for (int i = (size / 2) - 1; i >= 0; i--) {
            heapifyDown(i);
        }
    }
    
    /**
     * Heapify up (bubble up)
     * Time Complexity: O(log n)
     */
    private void heapifyUp(int index) {
        while (index > 0) {
            int parentIndex = getParentIndex(index);
            
            if (heap[index].compareTo(heap[parentIndex]) <= 0) {
                break;
            }
            
            swap(index, parentIndex);
            index = parentIndex;
        }
    }
    
    /**
     * Heapify down (bubble down)
     * Time Complexity: O(log n)
     */
    private void heapifyDown(int index) {
        while (getLeftChildIndex(index) < size) {
            int largerChildIndex = getLeftChildIndex(index);
            
            if (getRightChildIndex(index) < size && 
                heap[getRightChildIndex(index)].compareTo(heap[largerChildIndex]) > 0) {
                largerChildIndex = getRightChildIndex(index);
            }
            
            if (heap[index].compareTo(heap[largerChildIndex]) >= 0) {
                break;
            }
            
            swap(index, largerChildIndex);
            index = largerChildIndex;
        }
    }
    
    /**
     * Heap sort algorithm (descending order for max heap)
     * Time Complexity: O(n log n)
     */
    public T[] heapSort() {
        @SuppressWarnings("unchecked")
        T[] sortedArray = (T[]) new Comparable[size];
        int originalSize = size;
        
        for (int i = 0; i < originalSize; i++) {
            sortedArray[i] = extractMax();
        }
        
        return sortedArray;
    }
    
    /**
     * Find k largest elements
     * Time Complexity: O(k log n)
     */
    public T[] findKLargest(int k) {
        if (k > size) {
            throw new IllegalArgumentException("k is greater than heap size");
        }
        
        @SuppressWarnings("unchecked")
        T[] result = (T[]) new Comparable[k];
        MaxHeap<T> tempHeap = new MaxHeap<>(capacity);
        
        // Copy heap
        for (int i = 0; i < size; i++) {
            tempHeap.insert(heap[i]);
        }
        
        // Extract k largest
        for (int i = 0; i < k; i++) {
            result[i] = tempHeap.extractMax();
        }
        
        return result;
    }
    
    /**
     * Merge two max heaps
     * Time Complexity: O(n + m)
     */
    public MaxHeap<T> merge(MaxHeap<T> other) {
        MaxHeap<T> merged = new MaxHeap<>(this.size + other.size);
        
        // Add all elements from both heaps
        for (int i = 0; i < this.size; i++) {
            merged.insert(this.heap[i]);
        }
        
        for (int i = 0; i < other.size; i++) {
            merged.insert(other.heap[i]);
        }
        
        return merged;
    }
    
    // Helper methods
    private int getParentIndex(int index) {
        return (index - 1) / 2;
    }
    
    private int getLeftChildIndex(int index) {
        return 2 * index + 1;
    }
    
    private int getRightChildIndex(int index) {
        return 2 * index + 2;
    }
    
    private void swap(int i, int j) {
        T temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }
    
    @SuppressWarnings("unchecked")
    private void resize() {
        capacity *= 2;
        T[] newHeap = (T[]) new Comparable[capacity];
        System.arraycopy(heap, 0, newHeap, 0, size);
        heap = newHeap;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    public int size() {
        return size;
    }
    
    public void display() {
        if (isEmpty()) {
            System.out.println("Heap is empty");
            return;
        }
        
        System.out.print("Max Heap: ");
        for (int i = 0; i < size; i++) {
            System.out.print(heap[i] + " ");
        }
        System.out.println();
        
        printHeapStructure();
    }
    
    private void printHeapStructure() {
        if (isEmpty()) return;
        
        System.out.println("Heap Structure:");
        int level = 0;
        int index = 0;
        
        while (index < size) {
            int levelSize = (int) Math.pow(2, level);
            int itemsInLevel = Math.min(levelSize, size - index);
            
            // Print level
            for (int i = 0; i < itemsInLevel; i++) {
                System.out.print(heap[index + i] + " ");
            }
            System.out.println();
            
            index += itemsInLevel;
            level++;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Max Heap Demo:");
        System.out.println("==============");
        
        MaxHeap<Integer> heap = new MaxHeap<>();
        
        // Insert elements
        System.out.println("Inserting: 10, 5, 20, 1, 15, 30, 8");
        heap.insert(10);
        heap.insert(5);
        heap.insert(20);
        heap.insert(1);
        heap.insert(15);
        heap.insert(30);
        heap.insert(8);
        
        heap.display();
        
        // Extract maximum
        System.out.println("\nExtracting maximum: " + heap.extractMax());
        heap.display();
        
        // Peek
        System.out.println("\nPeek maximum: " + heap.peek());
        
        // Find k largest elements
        System.out.println("\nFinding 3 largest elements:");
        Integer[] kLargest = heap.findKLargest(3);
        for (Integer num : kLargest) {
            System.out.print(num + " ");
        }
        System.out.println();
        
        // Build heap from array
        System.out.println("\nBuilding heap from array [25, 12, 7, 3, 8, 15, 20]:");
        Integer[] array = {25, 12, 7, 3, 8, 15, 20};
        MaxHeap<Integer> heapFromArray = new MaxHeap<>(array);
        heapFromArray.display();
        
        // Heap sort
        System.out.println("\nHeap sort result (descending):");
        Integer[] sorted = heapFromArray.heapSort();
        for (Integer num : sorted) {
            System.out.print(num + " ");
        }
        System.out.println();
        
        // Merge two heaps
        System.out.println("\nMerging two heaps:");
        MaxHeap<Integer> heap1 = new MaxHeap<>();
        heap1.insert(50);
        heap1.insert(40);
        heap1.insert(60);
        
        MaxHeap<Integer> heap2 = new MaxHeap<>();
        heap2.insert(35);
        heap2.insert(45);
        
        MaxHeap<Integer> merged = heap1.merge(heap2);
        merged.display();
    }
}
