package Algorithms.DataStructures.LinkedLists;

/**
 * Circular Linked List Implementation
 */
public class CircularLinkedList<T> {
    
    private Node<T> tail; // We keep reference to tail for efficient operations
    private int size;
    
    /**
     * Node class for circular linked list
     */
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public CircularLinkedList() {
        this.tail = null;
        this.size = 0;
    }
    
    /**
     * Add element at the beginning
     * Time Complexity: O(1)
     */
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (tail == null) {
            tail = newNode;
            tail.next = tail; // Point to itself
        } else {
            newNode.next = tail.next; // Point to head
            tail.next = newNode; // Tail points to new head
        }
        size++;
    }
    
    /**
     * Add element at the end
     * Time Complexity: O(1)
     */
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (tail == null) {
            tail = newNode;
            tail.next = tail; // Point to itself
        } else {
            newNode.next = tail.next; // Point to head
            tail.next = newNode; // Current tail points to new node
            tail = newNode; // Update tail
        }
        size++;
    }
    
    /**
     * Add element at specific index
     * Time Complexity: O(n)
     */
    public void addAt(int index, T data) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        
        if (index == 0) {
            addFirst(data);
            return;
        }
        
        if (index == size) {
            addLast(data);
            return;
        }
        
        Node<T> newNode = new Node<>(data);
        Node<T> current = tail.next; // Start from head
        
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        newNode.next = current.next;
        current.next = newNode;
        size++;
    }
    
    /**
     * Remove first element
     * Time Complexity: O(1)
     */
    public T removeFirst() {
        if (tail == null) {
            throw new RuntimeException("List is empty");
        }
        
        Node<T> head = tail.next;
        T data = head.data;
        
        if (head == tail) { // Only one element
            tail = null;
        } else {
            tail.next = head.next; // Skip the head
        }
        
        size--;
        return data;
    }
    
    /**
     * Remove last element
     * Time Complexity: O(n)
     */
    public T removeLast() {
        if (tail == null) {
            throw new RuntimeException("List is empty");
        }
        
        T data = tail.data;
        
        if (tail.next == tail) { // Only one element
            tail = null;
        } else {
            // Find the node before tail
            Node<T> current = tail.next;
            while (current.next != tail) {
                current = current.next;
            }
            current.next = tail.next; // Skip tail
            tail = current; // Update tail
        }
        
        size--;
        return data;
    }
    
    /**
     * Remove element at specific index
     * Time Complexity: O(n)
     */
    public T removeAt(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        
        if (index == 0) {
            return removeFirst();
        }
        
        Node<T> current = tail.next; // Start from head
        
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        T data = current.next.data;
        
        if (current.next == tail) { // Removing tail
            tail = current;
        }
        
        current.next = current.next.next;
        size--;
        return data;
    }
    
    /**
     * Get element at index
     * Time Complexity: O(n)
     */
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        
        Node<T> current = tail.next; // Start from head
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        return current.data;
    }
    
    /**
     * Find element in list
     * Time Complexity: O(n)
     */
    public boolean contains(T data) {
        if (tail == null) {
            return false;
        }
        
        Node<T> current = tail.next; // Start from head
        do {
            if (current.data.equals(data)) {
                return true;
            }
            current = current.next;
        } while (current != tail.next);
        
        return false;
    }
    
    /**
     * Rotate list by k positions
     * Time Complexity: O(n)
     */
    public void rotate(int k) {
        if (tail == null || size <= 1) {
            return;
        }
        
        k = k % size; // Handle k > size
        if (k == 0) {
            return;
        }
        
        // Find new tail (size - k - 1 steps from head)
        Node<T> current = tail.next; // Start from head
        for (int i = 0; i < size - k - 1; i++) {
            current = current.next;
        }
        
        tail = current;
    }
    
    /**
     * Split circular list into two halves
     * Time Complexity: O(n)
     */
    public CircularLinkedList<T> split() {
        if (tail == null || size <= 1) {
            return new CircularLinkedList<>();
        }
        
        // Find middle point
        Node<T> slow = tail.next;
        Node<T> fast = tail.next;
        
        while (fast.next != tail.next && fast.next.next != tail.next) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // Create second list
        CircularLinkedList<T> secondList = new CircularLinkedList<>();
        secondList.tail = tail;
        secondList.size = size / 2 + (size % 2);
        
        // Update first list
        tail = slow;
        tail.next = secondList.tail.next; // Close the circle
        size = size / 2;
        
        // Update second list
        secondList.tail.next = slow.next;
        
        return secondList;
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Display the circular linked list
     */
    public void display() {
        if (tail == null) {
            System.out.println("List is empty");
            return;
        }
        
        Node<T> current = tail.next; // Start from head
        System.out.print("Circular List: ");
        do {
            System.out.print(current.data);
            current = current.next;
            if (current != tail.next) {
                System.out.print(" -> ");
            }
        } while (current != tail.next);
        System.out.println(" -> (back to start)");
    }
    
    /**
     * Display n traversals around the circle
     */
    public void displayTraversals(int traversals) {
        if (tail == null) {
            System.out.println("List is empty");
            return;
        }
        
        Node<T> current = tail.next; // Start from head
        System.out.print("Traversals: ");
        
        for (int i = 0; i < traversals * size; i++) {
            System.out.print(current.data);
            current = current.next;
            if (i < traversals * size - 1) {
                System.out.print(" -> ");
            }
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Circular Linked List Demo:");
        System.out.println("===========================");
        
        CircularLinkedList<Integer> list = new CircularLinkedList<>();
        
        // Add elements
        list.addFirst(10);
        list.addFirst(5);
        list.addLast(20);
        list.addLast(30);
        list.addAt(2, 15);
        
        System.out.println("List after additions:");
        list.display();
        System.out.println("Size: " + list.size());
        
        // Show circular nature
        System.out.println("\nCircular traversal (2 complete rounds):");
        list.displayTraversals(2);
        
        // Access elements
        System.out.println("\nAccess operations:");
        System.out.println("Element at index 2: " + list.get(2));
        System.out.println("Contains 15: " + list.contains(15));
        
        // Remove elements
        System.out.println("\nRemove operations:");
        System.out.println("Removed first: " + list.removeFirst());
        System.out.println("Removed last: " + list.removeLast());
        
        System.out.println("List after removals:");
        list.display();
        
        // Rotate list
        System.out.println("\nRotating list by 1 position:");
        list.rotate(1);
        list.display();
        
        // Split list
        System.out.println("\nSplitting list:");
        CircularLinkedList<Integer> secondHalf = list.split();
        System.out.println("First half:");
        list.display();
        System.out.println("Second half:");
        secondHalf.display();
    }
}
