package Algorithms.DataStructures.Queues;

/**
 * Linked List-based Queue Implementation
 */
public class LinkedQueue<T> {
    
    private Node<T> front;
    private Node<T> rear;
    private int size;
    
    /**
     * Node class for queue
     */
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public LinkedQueue() {
        this.front = null;
        this.rear = null;
        this.size = 0;
    }
    
    /**
     * Add element to rear of queue
     * Time Complexity: O(1)
     */
    public void enqueue(T item) {
        Node<T> newNode = new Node<>(item);
        
        if (rear == null) {
            front = rear = newNode;
        } else {
            rear.next = newNode;
            rear = newNode;
        }
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
        
        T data = front.data;
        front = front.next;
        
        if (front == null) {
            rear = null;
        }
        
        size--;
        return data;
    }
    
    /**
     * Peek at front element without removing
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        return front.data;
    }
    
    /**
     * Check if queue is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return front == null;
    }
    
    /**
     * Get size of queue
     * Time Complexity: O(1)
     */
    public int size() {
        return size;
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
        Node<T> current = front;
        while (current != null) {
            System.out.print(current.data);
            current = current.next;
            if (current != null) {
                System.out.print(" <- ");
            }
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Linked Queue Demo:");
        System.out.println("==================");
        
        LinkedQueue<String> queue = new LinkedQueue<>();
        
        // Enqueue elements
        System.out.println("Enqueuing elements: A, B, C, D, E");
        queue.enqueue("A");
        queue.enqueue("B");
        queue.enqueue("C");
        queue.enqueue("D");
        queue.enqueue("E");
        
        queue.display();
        System.out.println("Size: " + queue.size());
        
        // Peek and dequeue operations
        System.out.println("\nPeek: " + queue.peek());
        System.out.println("Dequeue: " + queue.dequeue());
        System.out.println("Dequeue: " + queue.dequeue());
        
        queue.display();
        System.out.println("Size: " + queue.size());
        System.out.println("Is empty: " + queue.isEmpty());
        
        // Clear queue
        System.out.println("\nClearing queue...");
        queue.clear();
        System.out.println("Is empty: " + queue.isEmpty());
    }
}
