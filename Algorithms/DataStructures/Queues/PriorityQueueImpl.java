package Algorithms.DataStructures.Queues;

import java.util.Collections;
import java.util.PriorityQueue;

/**
 * Priority Queue Implementation and Algorithms
 */
public class PriorityQueueImpl<T extends Comparable<T>> {
    
    private T[] heap;
    private int size;
    private int capacity;
    private boolean isMaxHeap;
    
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public PriorityQueueImpl() {
        this(DEFAULT_CAPACITY, false); // Min heap by default
    }
    
    @SuppressWarnings("unchecked")
    public PriorityQueueImpl(int capacity, boolean isMaxHeap) {
        this.capacity = capacity;
        this.heap = (T[]) new Comparable[capacity];
        this.size = 0;
        this.isMaxHeap = isMaxHeap;
    }
    
    /**
     * Insert element into priority queue
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
     * Remove and return highest priority element
     * Time Complexity: O(log n)
     */
    public T extractTop() {
        if (isEmpty()) {
            throw new RuntimeException("Priority queue is empty");
        }
        
        T top = heap[0];
        heap[0] = heap[size - 1];
        heap[size - 1] = null;
        size--;
        
        if (size > 0) {
            heapifyDown(0);
        }
        
        return top;
    }
    
    /**
     * Peek at highest priority element
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new RuntimeException("Priority queue is empty");
        }
        return heap[0];
    }
    
    /**
     * Check if priority queue is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Get size of priority queue
     * Time Complexity: O(1)
     */
    public int size() {
        return size;
    }
    
    /**
     * Heapify up (bubble up)
     * Time Complexity: O(log n)
     */
    private void heapifyUp(int index) {
        while (index > 0) {
            int parentIndex = (index - 1) / 2;
            
            if (compare(heap[index], heap[parentIndex]) <= 0) {
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
        while (true) {
            int largest = index;
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            
            if (leftChild < size && compare(heap[leftChild], heap[largest]) > 0) {
                largest = leftChild;
            }
            
            if (rightChild < size && compare(heap[rightChild], heap[largest]) > 0) {
                largest = rightChild;
            }
            
            if (largest == index) {
                break;
            }
            
            swap(index, largest);
            index = largest;
        }
    }
    
    /**
     * Compare two elements based on heap type
     */
    private int compare(T a, T b) {
        return isMaxHeap ? a.compareTo(b) : b.compareTo(a);
    }
    
    /**
     * Swap two elements in heap
     */
    private void swap(int i, int j) {
        T temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }
    
    /**
     * Resize array when capacity is reached
     */
    @SuppressWarnings("unchecked")
    private void resize() {
        capacity *= 2;
        T[] newHeap = (T[]) new Comparable[capacity];
        System.arraycopy(heap, 0, newHeap, 0, size);
        heap = newHeap;
    }
    
    /**
     * Display heap contents
     */
    public void display() {
        if (isEmpty()) {
            System.out.println("Priority queue is empty");
            return;
        }
        
        System.out.print("Priority Queue: ");
        for (int i = 0; i < size; i++) {
            System.out.print(heap[i] + " ");
        }
        System.out.println();
    }
    
    /**
     * Deque Implementation using two stacks
     */
    public static class Deque<T> {
        private java.util.Stack<T> stack1; // For front operations
        private java.util.Stack<T> stack2; // For rear operations
        
        public Deque() {
            stack1 = new java.util.Stack<>();
            stack2 = new java.util.Stack<>();
        }
        
        public void addFront(T item) {
            stack1.push(item);
        }
        
        public void addRear(T item) {
            stack2.push(item);
        }
        
        public T removeFront() {
            if (stack1.isEmpty()) {
                if (stack2.isEmpty()) {
                    throw new RuntimeException("Deque is empty");
                }
                while (!stack2.isEmpty()) {
                    stack1.push(stack2.pop());
                }
            }
            return stack1.pop();
        }
        
        public T removeRear() {
            if (stack2.isEmpty()) {
                if (stack1.isEmpty()) {
                    throw new RuntimeException("Deque is empty");
                }
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();
        }
        
        public boolean isEmpty() {
            return stack1.isEmpty() && stack2.isEmpty();
        }
        
        public int size() {
            return stack1.size() + stack2.size();
        }
    }
    
    /**
     * Queue using two stacks
     */
    public static class QueueUsingStacks<T> {
        private java.util.Stack<T> stack1; // For enqueue
        private java.util.Stack<T> stack2; // For dequeue
        
        public QueueUsingStacks() {
            stack1 = new java.util.Stack<>();
            stack2 = new java.util.Stack<>();
        }
        
        public void enqueue(T item) {
            stack1.push(item);
        }
        
        public T dequeue() {
            if (stack2.isEmpty()) {
                if (stack1.isEmpty()) {
                    throw new RuntimeException("Queue is empty");
                }
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();
        }
        
        public T peek() {
            if (stack2.isEmpty()) {
                if (stack1.isEmpty()) {
                    throw new RuntimeException("Queue is empty");
                }
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.peek();
        }
        
        public boolean isEmpty() {
            return stack1.isEmpty() && stack2.isEmpty();
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Priority Queue Demo:");
        System.out.println("====================");
        
        // Min heap
        PriorityQueueImpl<Integer> minHeap = new PriorityQueueImpl<>(10, false);
        System.out.println("Min Heap:");
        minHeap.insert(10);
        minHeap.insert(5);
        minHeap.insert(20);
        minHeap.insert(1);
        minHeap.insert(15);
        
        minHeap.display();
        System.out.println("Extract min: " + minHeap.extractTop());
        System.out.println("Extract min: " + minHeap.extractTop());
        minHeap.display();
        
        // Max heap
        PriorityQueueImpl<Integer> maxHeap = new PriorityQueueImpl<>(10, true);
        System.out.println("\nMax Heap:");
        maxHeap.insert(10);
        maxHeap.insert(5);
        maxHeap.insert(20);
        maxHeap.insert(1);
        maxHeap.insert(15);
        
        maxHeap.display();
        System.out.println("Extract max: " + maxHeap.extractTop());
        System.out.println("Extract max: " + maxHeap.extractTop());
        maxHeap.display();
        
        // Deque demo
        System.out.println("\nDeque Demo:");
        Deque<String> deque = new Deque<>();
        deque.addFront("B");
        deque.addFront("A");
        deque.addRear("C");
        deque.addRear("D");
        
        System.out.println("Remove front: " + deque.removeFront()); // A
        System.out.println("Remove rear: " + deque.removeRear());   // D
        System.out.println("Remove front: " + deque.removeFront()); // B
        
        // Queue using stacks demo
        System.out.println("\nQueue using Stacks Demo:");
        QueueUsingStacks<Integer> queue = new QueueUsingStacks<>();
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);
        
        System.out.println("Dequeue: " + queue.dequeue()); // 1
        System.out.println("Peek: " + queue.peek());       // 2
        System.out.println("Dequeue: " + queue.dequeue()); // 2
        
        // Java's built-in PriorityQueue example
        System.out.println("\nJava's PriorityQueue Demo:");
        PriorityQueue<Integer> javaPQ = new PriorityQueue<>();
        javaPQ.add(10);
        javaPQ.add(5);
        javaPQ.add(20);
        javaPQ.add(1);
        
        System.out.println("Min heap: " + javaPQ);
        System.out.println("Poll: " + javaPQ.poll());
        System.out.println("After poll: " + javaPQ);
        
        // Max heap using Collections.reverseOrder()
        PriorityQueue<Integer> maxPQ = new PriorityQueue<>(Collections.reverseOrder());
        maxPQ.add(10);
        maxPQ.add(5);
        maxPQ.add(20);
        maxPQ.add(1);
        
        System.out.println("Max heap: " + maxPQ);
        System.out.println("Poll: " + maxPQ.poll());
        System.out.println("After poll: " + maxPQ);
    }
}
