package Algorithms.DataStructures.Stacks;

import java.util.EmptyStackException;

/**
 * Array-based Stack Implementation
 */
public class ArrayStack<T> {
    
    private T[] stack;
    private int top;
    private int capacity;
    
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public ArrayStack() {
        this.capacity = DEFAULT_CAPACITY;
        this.stack = (T[]) new Object[capacity];
        this.top = -1;
    }
    
    @SuppressWarnings("unchecked")
    public ArrayStack(int capacity) {
        this.capacity = capacity;
        this.stack = (T[]) new Object[capacity];
        this.top = -1;
    }
    
    /**
     * Push element onto stack
     * Time Complexity: O(1) amortized
     */
    public void push(T item) {
        if (isFull()) {
            resize();
        }
        stack[++top] = item;
    }
    
    /**
     * Pop element from stack
     * Time Complexity: O(1)
     */
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        T item = stack[top];
        stack[top--] = null; // Help GC
        return item;
    }
    
    /**
     * Peek at top element without removing
     * Time Complexity: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return stack[top];
    }
    
    /**
     * Check if stack is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return top == -1;
    }
    
    /**
     * Check if stack is full
     * Time Complexity: O(1)
     */
    public boolean isFull() {
        return top == capacity - 1;
    }
    
    /**
     * Get size of stack
     * Time Complexity: O(1)
     */
    public int size() {
        return top + 1;
    }
    
    /**
     * Resize array when capacity is reached
     * Time Complexity: O(n)
     */
    @SuppressWarnings("unchecked")
    private void resize() {
        capacity *= 2;
        T[] newStack = (T[]) new Object[capacity];
        System.arraycopy(stack, 0, newStack, 0, top + 1);
        stack = newStack;
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
        for (int i = top; i >= 0; i--) {
            System.out.print(stack[i]);
            if (i > 0) {
                System.out.print(" | ");
            }
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        System.out.println("Array Stack Demo:");
        System.out.println("=================");
        
        ArrayStack<Integer> stack = new ArrayStack<>(5);
        
        // Push elements
        System.out.println("Pushing elements: 10, 20, 30, 40, 50");
        stack.push(10);
        stack.push(20);
        stack.push(30);
        stack.push(40);
        stack.push(50);
        
        stack.display();
        System.out.println("Size: " + stack.size());
        System.out.println("Is full: " + stack.isFull());
        
        // Push one more to trigger resize
        System.out.println("\nPushing 60 (triggers resize):");
        stack.push(60);
        stack.display();
        
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
