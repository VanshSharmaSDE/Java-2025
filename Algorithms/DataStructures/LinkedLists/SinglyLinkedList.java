package Algorithms.DataStructures.LinkedLists;

/**
 * Singly Linked List Implementation
 */
public class SinglyLinkedList<T> {
    
    private Node<T> head;
    private int size;
    
    /**
     * Node class for linked list
     */
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public SinglyLinkedList() {
        this.head = null;
        this.size = 0;
    }
    
    /**
     * Add element at the beginning
     * Time Complexity: O(1)
     */
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        newNode.next = head;
        head = newNode;
        size++;
    }
    
    /**
     * Add element at the end
     * Time Complexity: O(n)
     */
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = newNode;
        } else {
            Node<T> current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
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
        
        Node<T> newNode = new Node<>(data);
        Node<T> current = head;
        
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
        if (head == null) {
            throw new RuntimeException("List is empty");
        }
        
        T data = head.data;
        head = head.next;
        size--;
        return data;
    }
    
    /**
     * Remove last element
     * Time Complexity: O(n)
     */
    public T removeLast() {
        if (head == null) {
            throw new RuntimeException("List is empty");
        }
        
        if (head.next == null) {
            T data = head.data;
            head = null;
            size--;
            return data;
        }
        
        Node<T> current = head;
        while (current.next.next != null) {
            current = current.next;
        }
        
        T data = current.next.data;
        current.next = null;
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
        
        Node<T> current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        T data = current.next.data;
        current.next = current.next.next;
        size--;
        return data;
    }
    
    /**
     * Find element in list
     * Time Complexity: O(n)
     */
    public boolean contains(T data) {
        Node<T> current = head;
        while (current != null) {
            if (current.data.equals(data)) {
                return true;
            }
            current = current.next;
        }
        return false;
    }
    
    /**
     * Get element at index
     * Time Complexity: O(n)
     */
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        
        Node<T> current = head;
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        return current.data;
    }
    
    /**
     * Reverse the linked list
     * Time Complexity: O(n)
     */
    public void reverse() {
        Node<T> prev = null;
        Node<T> current = head;
        Node<T> next;
        
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        head = prev;
    }
    
    /**
     * Find middle element (Floyd's Cycle Detection)
     * Time Complexity: O(n)
     */
    public T findMiddle() {
        if (head == null) {
            throw new RuntimeException("List is empty");
        }
        
        Node<T> slow = head;
        Node<T> fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow.data;
    }
    
    /**
     * Detect cycle in linked list (Floyd's Algorithm)
     * Time Complexity: O(n)
     */
    public boolean hasCycle() {
        if (head == null || head.next == null) {
            return false;
        }
        
        Node<T> slow = head;
        Node<T> fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) {
                return true;
            }
        }
        
        return false;
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Display the linked list
     */
    public void display() {
        if (head == null) {
            System.out.println("List is empty");
            return;
        }
        
        Node<T> current = head;
        while (current != null) {
            System.out.print(current.data);
            if (current.next != null) {
                System.out.print(" -> ");
            }
            current = current.next;
        }
        System.out.println(" -> null");
    }
    
    public static void main(String[] args) {
        System.out.println("Singly Linked List Demo:");
        System.out.println("========================");
        
        SinglyLinkedList<Integer> list = new SinglyLinkedList<>();
        
        // Add elements
        list.addFirst(10);
        list.addFirst(5);
        list.addLast(20);
        list.addLast(30);
        list.addAt(2, 15);
        
        System.out.println("List after additions:");
        list.display();
        System.out.println("Size: " + list.size());
        
        // Find operations
        System.out.println("\nFind operations:");
        System.out.println("Contains 15: " + list.contains(15));
        System.out.println("Element at index 2: " + list.get(2));
        System.out.println("Middle element: " + list.findMiddle());
        
        // Remove operations
        System.out.println("\nRemove operations:");
        System.out.println("Removed first: " + list.removeFirst());
        System.out.println("Removed last: " + list.removeLast());
        System.out.println("Removed at index 1: " + list.removeAt(1));
        
        System.out.println("List after removals:");
        list.display();
        
        // Reverse list
        System.out.println("\nReversing list:");
        list.reverse();
        list.display();
        
        // Cycle detection
        System.out.println("Has cycle: " + list.hasCycle());
    }
}
