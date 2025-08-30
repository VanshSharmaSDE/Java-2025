package Algorithms.DataStructures.Queues;

/**
 * Array-based Queue Implementation (Circular Queue)
 */
public class ArrayQueue<T> {
    
    private T[] queue;
    private int front;
    private int rear;
    private int size;
    private int capacity;
    
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public ArrayQueue() {
        this.capacity = DEFAULT_CAPACITY;
        this.queue = (T[]) new Object[capacity];
        this.front = 0;
        this.rear = -1;
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public ArrayQueue(int capacity) {
        this.capacity = capacity;
        this.queue = (T[]) new Object[capacity];
        this.front = 0;
        this.rear = -1;
        this.size = 0;
    }
    
    /**
     * Add element to rear of queue
     * Time Complexity: O(1) amortized
     */
    public void enqueue(T item) {
        if (isFull()) {
            resize();
        }
        rear = (rear + 1) % capacity;
        queue[rear] = item;
        size++;
    }
    
    /**
     * Remove element from front of queue
     * Time Complexity: O(1)
     */
    public T dequeue() {
        if (isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        T item = queue[front];
        queue[front] = null; // Help GC
        front = (front + 1) % capacity;
        size--;
        return item;
    }
    
    /**
     * Peek at front element without removing
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        return queue[front];
    }
    
    /**
     * Check if queue is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Check if queue is full
     * Time Complexity: O(1)
     */
    public boolean isFull() {
        return size == capacity;
    }
    
    /**
     * Get size of queue
     * Time Complexity: O(1)
     */
    public int size() {
        return size;
    }
    
    /**
     * Resize array when capacity is reached
     * Time Complexity: O(n)
     */
    @SuppressWarnings("unchecked")
    private void resize() {
        T[] newQueue = (T[]) new Object[capacity * 2];
        
        for (int i = 0; i < size; i++) {
            newQueue[i] = queue[(front + i) % capacity];
        }
        
        queue = newQueue;
        front = 0;
        rear = size - 1;
        capacity *= 2;
    }
    
    /**
     * Clear all elements from queue
     * Time Complexity: O(n)
     */
    public void clear() {
        while (!isEmpty()) {
            dequeue();
        }
    }
    
    /**
     * Display queue contents
     */
    public void display() {
        if (isEmpty()) {
            System.out.println("Queue is empty");
            return;
        }
        
        System.out.print("Queue (front to rear): ");
        for (int i = 0; i < size; i++) {
            System.out.print(queue[(front + i) % capacity]);
            if (i < size - 1) {
                System.out.print(" <- ");
            }
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Array Queue Demo:");
        System.out.println("=================");
        
        ArrayQueue<Integer> queue = new ArrayQueue<>(5);
        
        // Enqueue elements
        System.out.println("Enqueuing elements: 10, 20, 30, 40, 50");
        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);
        queue.enqueue(40);
        queue.enqueue(50);
        
        queue.display();
        System.out.println("Size: " + queue.size());
        System.out.println("Is full: " + queue.isFull());
        
        // Enqueue one more to trigger resize
        System.out.println("\nEnqueuing 60 (triggers resize):");
        queue.enqueue(60);
        queue.display();
        
        // Peek and dequeue operations
        System.out.println("\nPeek: " + queue.peek());
        System.out.println("Dequeue: " + queue.dequeue());
        System.out.println("Dequeue: " + queue.dequeue());
        
        queue.display();
        System.out.println("Size: " + queue.size());
        
        // Test circular nature
        System.out.println("\nEnqueuing 70, 80:");
        queue.enqueue(70);
        queue.enqueue(80);
        queue.display();
        
        // Clear queue
        System.out.println("\nClearing queue...");
        queue.clear();
        System.out.println("Is empty: " + queue.isEmpty());
    }
}
