package Algorithms.DataStructures.Heaps;

/**
 * Min Heap Implementation with various heap algorithms
 */
public class MinHeap<T extends Comparable<T>> {
    
    private T[] heap;
    private int size;
    private int capacity;
    
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public MinHeap() {
        this.capacity = DEFAULT_CAPACITY;
        this.heap = (T[]) new Comparable[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public MinHeap(int capacity) {
        this.capacity = capacity;
        this.heap = (T[]) new Comparable[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public MinHeap(T[] array) {
        this.capacity = array.length;
        this.size = array.length;
        this.heap = (T[]) new Comparable[capacity];
        System.arraycopy(array, 0, heap, 0, size);
        buildHeap();
    }
    
    /**
     * Insert element into min heap
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
     * Extract minimum element (root)
     * Time Complexity: O(log n)
     */
    public T extractMin() {
        if (isEmpty()) {
            throw new RuntimeException("Heap is empty");
        }
        
        T min = heap[0];
        heap[0] = heap[size - 1];
        heap[size - 1] = null;
        size--;
        
        if (size > 0) {
            heapifyDown(0);
        }
        
        return min;
    }
    
    /**
     * Peek at minimum element without removing
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
     * Decrease key at specific index
     * Time Complexity: O(log n)
     */
    public void decreaseKey(int index, T newValue) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (newValue.compareTo(heap[index]) > 0) {
            throw new IllegalArgumentException("New value is greater than current value");
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
            
            if (heap[index].compareTo(heap[parentIndex]) >= 0) {
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
            int smallerChildIndex = getLeftChildIndex(index);
            
            if (getRightChildIndex(index) < size && 
                heap[getRightChildIndex(index)].compareTo(heap[smallerChildIndex]) < 0) {
                smallerChildIndex = getRightChildIndex(index);
            }
            
            if (heap[index].compareTo(heap[smallerChildIndex]) <= 0) {
                break;
            }
            
            swap(index, smallerChildIndex);
            index = smallerChildIndex;
        }
    }
    
    /**
     * Heap sort algorithm
     * Time Complexity: O(n log n)
     */
    public T[] heapSort() {
        @SuppressWarnings("unchecked")
        T[] sortedArray = (T[]) new Comparable[size];
        int originalSize = size;
        
        for (int i = 0; i < originalSize; i++) {
            sortedArray[i] = extractMin();
        }
        
        return sortedArray;
    }
    
    /**
     * Find k smallest elements
     * Time Complexity: O(k log n)
     */
    public T[] findKSmallest(int k) {
        if (k > size) {
            throw new IllegalArgumentException("k is greater than heap size");
        }
        
        @SuppressWarnings("unchecked")
        T[] result = (T[]) new Comparable[k];
        MinHeap<T> tempHeap = new MinHeap<>(capacity);
        
        // Copy heap
        for (int i = 0; i < size; i++) {
            tempHeap.insert(heap[i]);
        }
        
        // Extract k smallest
        for (int i = 0; i < k; i++) {
            result[i] = tempHeap.extractMin();
        }
        
        return result;
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
        
        System.out.print("Min Heap: ");
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
        System.out.println("Min Heap Demo:");
        System.out.println("==============");
        
        MinHeap<Integer> heap = new MinHeap<>();
        
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
        
        // Extract minimum
        System.out.println("\nExtracting minimum: " + heap.extractMin());
        heap.display();
        
        // Peek
        System.out.println("\nPeek minimum: " + heap.peek());
        
        // Find k smallest elements
        System.out.println("\nFinding 3 smallest elements:");
        Integer[] kSmallest = heap.findKSmallest(3);
        for (Integer num : kSmallest) {
            System.out.print(num + " ");
        }
        System.out.println();
        
        // Build heap from array
        System.out.println("\nBuilding heap from array [25, 12, 7, 3, 8, 15, 20]:");
        Integer[] array = {25, 12, 7, 3, 8, 15, 20};
        MinHeap<Integer> heapFromArray = new MinHeap<>(array);
        heapFromArray.display();
        
        // Heap sort
        System.out.println("\nHeap sort result:");
        Integer[] sorted = heapFromArray.heapSort();
        for (Integer num : sorted) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}
