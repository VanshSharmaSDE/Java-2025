package Algorithms.DataStructures.Stacks;

import java.util.EmptyStackException;

/**
 * Linked List-based Stack Implementation
 */
public class LinkedStack<T> {
    
    private Node<T> top;
    private int size;
    
    /**
     * Node class for stack
     */
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public LinkedStack() {
        this.top = null;
        this.size = 0;
    }
    
    /**
     * Push element onto stack
     * Time Complexity: O(1)
     */
    public void push(T item) {
        Node<T> newNode = new Node<>(item);
        newNode.next = top;
        top = newNode;
        size++;
    }
    
    /**
     * Pop element from stack
     * Time Complexity: O(1)
     */
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        T data = top.data;
        top = top.next;
        size--;
        return data;
    }
    
    /**
     * Peek at top element without removing
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return top.data;
    }
    
    /**
     * Check if stack is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return top == null;
    }
    
    /**
     * Get size of stack
     * Time Complexity: O(1)
     */
    public int size() {
        return size;
    }
    
    /**
     * Clear all elements from stack
     * Time Complexity: O(n)
     */
    public void clear() {
        while (!isEmpty()) {
            pop();
        }
    }
    
    /**
     * Display stack contents
     */
    public void display() {
        if (isEmpty()) {
            System.out.println("Stack is empty");
            return;
        }
        
        System.out.print("Stack (top to bottom): ");
        Node<T> current = top;
        while (current != null) {
            System.out.print(current.data);
            current = current.next;
            if (current != null) {
                System.out.print(" | ");
            }
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Linked Stack Demo:");
        System.out.println("==================");
        
        LinkedStack<String> stack = new LinkedStack<>();
        
        // Push elements
        System.out.println("Pushing elements: A, B, C, D, E");
        stack.push("A");
        stack.push("B");
        stack.push("C");
        stack.push("D");
        stack.push("E");
        
        stack.display();
        System.out.println("Size: " + stack.size());
        
        // Peek and pop operations
        System.out.println("\nPeek: " + stack.peek());
        System.out.println("Pop: " + stack.pop());
        System.out.println("Pop: " + stack.pop());
        
        stack.display();
        System.out.println("Size: " + stack.size());
        System.out.println("Is empty: " + stack.isEmpty());
        
        // Clear stack
        System.out.println("\nClearing stack...");
        stack.clear();
        System.out.println("Is empty: " + stack.isEmpty());
    }
}
