package Algorithms.DataStructures.HashTables;

/**
 * Separate Chaining Hash Table Implementation
 */
public class ChainingHashTable<K, V> {
    
    private static class Node<K, V> {
        K key;
        V value;
        Node<K, V> next;
        
        Node(K key, V value) {
            this.key = key;
            this.value = value;
            this.next = null;
        }
    }
    
    private Node<K, V>[] table;
    private int size;
    private int capacity;
    private double loadFactor;
    
    private static final int DEFAULT_CAPACITY = 16;
    private static final double DEFAULT_LOAD_FACTOR = 0.75;
    
    @SuppressWarnings("unchecked")
    public ChainingHashTable() {
        this.capacity = DEFAULT_CAPACITY;
        this.loadFactor = DEFAULT_LOAD_FACTOR;
        this.table = new Node[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public ChainingHashTable(int capacity, double loadFactor) {
        this.capacity = capacity;
        this.loadFactor = loadFactor;
        this.table = new Node[capacity];
        this.size = 0;
    }
    
    /**
     * Hash function using division method
     * Time Complexity: O(1)
     */
    private int hash(K key) {
        if (key == null) return 0;
        return Math.abs(key.hashCode()) % capacity;
    }
    
    /**
     * Put key-value pair in hash table
     * Time Complexity: O(1) average, O(n) worst case
     */
    public void put(K key, V value) {
        int index = hash(key);
        Node<K, V> head = table[index];
        
        // Search for existing key
        Node<K, V> current = head;
        while (current != null) {
            if (current.key.equals(key)) {
                current.value = value; // Update existing
                return;
            }
            current = current.next;
        }
        
        // Add new node at beginning of chain
        Node<K, V> newNode = new Node<>(key, value);
        newNode.next = head;
        table[index] = newNode;
        size++;
        
        // Check if resize is needed
        if (size > capacity * loadFactor) {
            resize();
        }
    }
    
    /**
     * Get value by key
     * Time Complexity: O(1) average, O(n) worst case
     */
    public V get(K key) {
        int index = hash(key);
        Node<K, V> current = table[index];
        
        while (current != null) {
            if (current.key.equals(key)) {
                return current.value;
            }
            current = current.next;
        }
        
        return null; // Key not found
    }
    
    /**
     * Remove key-value pair
     * Time Complexity: O(1) average, O(n) worst case
     */
    public V remove(K key) {
        int index = hash(key);
        Node<K, V> current = table[index];
        Node<K, V> prev = null;
        
        while (current != null) {
            if (current.key.equals(key)) {
                if (prev == null) {
                    table[index] = current.next; // Remove head
                } else {
                    prev.next = current.next;
                }
                size--;
                return current.value;
            }
            prev = current;
            current = current.next;
        }
        
        return null; // Key not found
    }
    
    /**
     * Check if key exists
     * Time Complexity: O(1) average, O(n) worst case
     */
    public boolean containsKey(K key) {
        return get(key) != null;
    }
    
    /**
     * Check if hash table is empty
     * Time Complexity: O(1)
     */
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Get size of hash table
     * Time Complexity: O(1)
     */
    public int size() {
        return size;
    }
    
    /**
     * Resize hash table when load factor exceeds threshold
     * Time Complexity: O(n)
     */
    @SuppressWarnings("unchecked")
    private void resize() {
        Node<K, V>[] oldTable = table;
        int oldCapacity = capacity;
        
        capacity *= 2;
        table = new Node[capacity];
        size = 0;
        
        // Rehash all elements
        for (int i = 0; i < oldCapacity; i++) {
            Node<K, V> current = oldTable[i];
            while (current != null) {
                put(current.key, current.value);
                current = current.next;
            }
        }
    }
    
    /**
     * Get all keys
     * Time Complexity: O(n)
     */
    public java.util.List<K> keySet() {
        java.util.List<K> keys = new java.util.ArrayList<>();
        
        for (int i = 0; i < capacity; i++) {
            Node<K, V> current = table[i];
            while (current != null) {
                keys.add(current.key);
                current = current.next;
            }
        }
        
        return keys;
    }
    
    /**
     * Get all values
     * Time Complexity: O(n)
     */
    public java.util.List<V> values() {
        java.util.List<V> vals = new java.util.ArrayList<>();
        
        for (int i = 0; i < capacity; i++) {
            Node<K, V> current = table[i];
            while (current != null) {
                vals.add(current.value);
                current = current.next;
            }
        }
        
        return vals;
    }
    
    /**
     * Clear all elements
     * Time Complexity: O(n)
     */
    @SuppressWarnings("unchecked")
    public void clear() {
        table = new Node[capacity];
        size = 0;
    }
    
    /**
     * Get current load factor
     */
    public double getCurrentLoadFactor() {
        return (double) size / capacity;
    }
    
    /**
     * Display hash table structure
     */
    public void display() {
        System.out.println("Hash Table (Chaining):");
        System.out.println("Size: " + size + ", Capacity: " + capacity);
        System.out.println("Load Factor: " + getCurrentLoadFactor());
        
        for (int i = 0; i < capacity; i++) {
            System.out.print("Bucket " + i + ": ");
            Node<K, V> current = table[i];
            if (current == null) {
                System.out.println("empty");
            } else {
                while (current != null) {
                    System.out.print("[" + current.key + ":" + current.value + "]");
                    current = current.next;
                    if (current != null) {
                        System.out.print(" -> ");
                    }
                }
                System.out.println();
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Chaining Hash Table Demo:");
        System.out.println("=========================");
        
        ChainingHashTable<String, Integer> hashTable = new ChainingHashTable<>(8, 0.75);
        
        // Insert key-value pairs
        System.out.println("Inserting key-value pairs:");
        hashTable.put("apple", 10);
        hashTable.put("banana", 20);
        hashTable.put("cherry", 30);
        hashTable.put("date", 40);
        hashTable.put("elderberry", 50);
        hashTable.put("fig", 60);
        hashTable.put("grape", 70);
        
        hashTable.display();
        
        // Get values
        System.out.println("\nGet operations:");
        System.out.println("apple: " + hashTable.get("apple"));
        System.out.println("banana: " + hashTable.get("banana"));
        System.out.println("nonexistent: " + hashTable.get("nonexistent"));
        
        // Update value
        System.out.println("\nUpdating apple to 100:");
        hashTable.put("apple", 100);
        System.out.println("apple: " + hashTable.get("apple"));
        
        // Remove key
        System.out.println("\nRemoving banana:");
        Integer removed = hashTable.remove("banana");
        System.out.println("Removed value: " + removed);
        System.out.println("banana: " + hashTable.get("banana"));
        
        // Check containment
        System.out.println("\nContainment checks:");
        System.out.println("Contains apple: " + hashTable.containsKey("apple"));
        System.out.println("Contains banana: " + hashTable.containsKey("banana"));
        
        // Display keys and values
        System.out.println("\nAll keys: " + hashTable.keySet());
        System.out.println("All values: " + hashTable.values());
        
        // Insert more to trigger resize
        System.out.println("\nInserting more items to trigger resize:");
        hashTable.put("honeydew", 80);
        hashTable.put("kiwi", 90);
        hashTable.put("lemon", 100);
        
        hashTable.display();
    }
}
