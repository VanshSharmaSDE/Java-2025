package Algorithms.DataStructures.LinkedLists;

/**
 * Doubly Linked List Implementation
 */
public class DoublyLinkedList<T> {
    
    private Node<T> head;
    private Node<T> tail;
    private int size;
    
    /**
     * Node class for doubly linked list
     */
    private static class Node<T> {
        T data;
        Node<T> next;
        Node<T> prev;
        
        Node(T data) {
            this.data = data;
            this.next = null;
            this.prev = null;
        }
    }
    
    public DoublyLinkedList() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    
    /**
     * Add element at the beginning
     * Time Complexity: O(1)
     */
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = tail = newNode;
        } else {
            newNode.next = head;
            head.prev = newNode;
            head = newNode;
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
            head = tail = newNode;
        } else {
            tail.next = newNode;
            newNode.prev = tail;
            tail = newNode;
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
        Node<T> current = getNodeAt(index);
        
        newNode.next = current;
        newNode.prev = current.prev;
        current.prev.next = newNode;
        current.prev = newNode;
        
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
        
        if (head == tail) {
            head = tail = null;
        } else {
            head = head.next;
            head.prev = null;
        }
        
        size--;
        return data;
    }
    
    /**
     * Remove last element
     * Time Complexity: O(1)
     */
    public T removeLast() {
        if (tail == null) {
            throw new RuntimeException("List is empty");
        }
        
        T data = tail.data;
        
        if (head == tail) {
            head = tail = null;
        } else {
            tail = tail.prev;
            tail.next = null;
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
        
        if (index == size - 1) {
            return removeLast();
        }
        
        Node<T> nodeToRemove = getNodeAt(index);
        T data = nodeToRemove.data;
        
        nodeToRemove.prev.next = nodeToRemove.next;
        nodeToRemove.next.prev = nodeToRemove.prev;
        
        size--;
        return data;
    }
    
    /**
     * Get node at specific index (optimized for doubly linked list)
     * Time Complexity: O(n/2)
     */
    private Node<T> getNodeAt(int index) {
        if (index < size / 2) {
            // Search from head
            Node<T> current = head;
            for (int i = 0; i < index; i++) {
                current = current.next;
            }
            return current;
        } else {
            // Search from tail
            Node<T> current = tail;
            for (int i = size - 1; i > index; i--) {
                current = current.prev;
            }
            return current;
        }
    }
    
    /**
     * Get element at index
     * Time Complexity: O(n/2)
     */
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        
        return getNodeAt(index).data;
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
     * Reverse the doubly linked list
     * Time Complexity: O(n)
     */
    public void reverse() {
        Node<T> current = head;
        Node<T> temp;
        
        while (current != null) {
            // Swap next and prev pointers
            temp = current.prev;
            current.prev = current.next;
            current.next = temp;
            
            // Move to the next node (which is actually prev now)
            current = current.prev;
        }
        
        // Swap head and tail
        temp = head;
        head = tail;
        tail = temp;
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Display the list forward
     */
    public void displayForward() {
        if (head == null) {
            System.out.println("List is empty");
            return;
        }
        
        System.out.print("Forward: null <- ");
        Node<T> current = head;
        while (current != null) {
            System.out.print(current.data);
            if (current.next != null) {
                System.out.print(" <-> ");
            }
            current = current.next;
        }
        System.out.println(" -> null");
    }
    
    /**
     * Display the list backward
     */
    public void displayBackward() {
        if (tail == null) {
            System.out.println("List is empty");
            return;
        }
        
        System.out.print("Backward: null <- ");
        Node<T> current = tail;
        while (current != null) {
            System.out.print(current.data);
            if (current.prev != null) {
                System.out.print(" <-> ");
            }
            current = current.prev;
        }
        System.out.println(" -> null");
    }
    
    public static void main(String[] args) {
        System.out.println("Doubly Linked List Demo:");
        System.out.println("=========================");
        
        DoublyLinkedList<String> list = new DoublyLinkedList<>();
        
        // Add elements
        list.addFirst("B");
        list.addFirst("A");
        list.addLast("C");
        list.addLast("D");
        list.addAt(2, "B.5");
        
        System.out.println("List after additions:");
        list.displayForward();
        list.displayBackward();
        System.out.println("Size: " + list.size());
        
        // Access elements
        System.out.println("\nAccess operations:");
        System.out.println("Element at index 2: " + list.get(2));
        System.out.println("Contains 'B.5': " + list.contains("B.5"));
        
        // Remove elements
        System.out.println("\nRemove operations:");
        System.out.println("Removed first: " + list.removeFirst());
        System.out.println("Removed last: " + list.removeLast());
        System.out.println("Removed at index 1: " + list.removeAt(1));
        
        System.out.println("List after removals:");
        list.displayForward();
        
        // Reverse list
        System.out.println("\nReversing list:");
        list.reverse();
        list.displayForward();
        list.displayBackward();
    }
}
