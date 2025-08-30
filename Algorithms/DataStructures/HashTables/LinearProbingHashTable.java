package Algorithms.DataStructures.HashTables;

/**
 * Open Addressing Hash Table Implementation (Linear Probing)
 */
public class LinearProbingHashTable<K, V> {
    
    private static class Entry<K, V> {
        K key;
        V value;
        boolean deleted; // For lazy deletion
        
        Entry(K key, V value) {
            this.key = key;
            this.value = value;
            this.deleted = false;
        }
    }
    
    private Entry<K, V>[] table;
    private int size;
    private int capacity;
    private double loadFactor;
    
    private static final int DEFAULT_CAPACITY = 16;
    private static final double DEFAULT_LOAD_FACTOR = 0.75;
    
    @SuppressWarnings("unchecked")
    public LinearProbingHashTable() {
        this.capacity = DEFAULT_CAPACITY;
        this.loadFactor = DEFAULT_LOAD_FACTOR;
        this.table = new Entry[capacity];
        this.size = 0;
    }
    
    @SuppressWarnings("unchecked")
    public LinearProbingHashTable(int capacity, double loadFactor) {
        this.capacity = capacity;
        this.loadFactor = loadFactor;
        this.table = new Entry[capacity];
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
     * Find index for key using linear probing
     * Time Complexity: O(1) average, O(n) worst case
     */
    private int findIndex(K key) {
        int index = hash(key);
        
        while (table[index] != null) {
            if (!table[index].deleted && table[index].key.equals(key)) {
                return index;
            }
            index = (index + 1) % capacity;
        }
        
        return -1; // Key not found
    }
    
    /**
     * Find empty slot for insertion using linear probing
     * Time Complexity: O(1) average, O(n) worst case
     */
    private int findEmptySlot(K key) {
        int index = hash(key);
        
        while (table[index] != null && !table[index].deleted) {
            if (table[index].key.equals(key)) {
                return index; // Key already exists
            }
            index = (index + 1) % capacity;
        }
        
        return index; // Empty or deleted slot found
    }
    
    /**
     * Put key-value pair in hash table
     * Time Complexity: O(1) average, O(n) worst case
     */
    public void put(K key, V value) {
        if (size >= capacity * loadFactor) {
            resize();
        }
        
        int index = findEmptySlot(key);
        
        if (table[index] == null || table[index].deleted) {
            table[index] = new Entry<>(key, value);
            size++;
        } else {
            // Update existing key
            table[index].value = value;
        }
    }
    
    /**
     * Get value by key
     * Time Complexity: O(1) average, O(n) worst case
     */
    public V get(K key) {
        int index = findIndex(key);
        return (index != -1) ? table[index].value : null;
    }
    
    /**
     * Remove key-value pair (lazy deletion)
     * Time Complexity: O(1) average, O(n) worst case
     */
    public V remove(K key) {
        int index = findIndex(key);
        
        if (index != -1) {
            V value = table[index].value;
            table[index].deleted = true;
            size--;
            return value;
        }
        
        return null; // Key not found
    }
    
    /**
     * Check if key exists
     * Time Complexity: O(1) average, O(n) worst case
     */
    public boolean containsKey(K key) {
        return findIndex(key) != -1;
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
        Entry<K, V>[] oldTable = table;
        int oldCapacity = capacity;
        
        capacity *= 2;
        table = new Entry[capacity];
        size = 0;
        
        // Rehash all non-deleted elements
        for (int i = 0; i < oldCapacity; i++) {
            if (oldTable[i] != null && !oldTable[i].deleted) {
                put(oldTable[i].key, oldTable[i].value);
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
            if (table[i] != null && !table[i].deleted) {
                keys.add(table[i].key);
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
            if (table[i] != null && !table[i].deleted) {
                vals.add(table[i].value);
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
        table = new Entry[capacity];
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
        System.out.println("Hash Table (Linear Probing):");
        System.out.println("Size: " + size + ", Capacity: " + capacity);
        System.out.println("Load Factor: " + getCurrentLoadFactor());
        
        for (int i = 0; i < capacity; i++) {
            System.out.print("Index " + i + ": ");
            if (table[i] == null) {
                System.out.println("empty");
            } else if (table[i].deleted) {
                System.out.println("deleted");
            } else {
                System.out.println("[" + table[i].key + ":" + table[i].value + "]");
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Linear Probing Hash Table Demo:");
        System.out.println("===============================");
        
        LinearProbingHashTable<String, Integer> hashTable = new LinearProbingHashTable<>(8, 0.75);
        
        // Insert key-value pairs
        System.out.println("Inserting key-value pairs:");
        hashTable.put("apple", 10);
        hashTable.put("banana", 20);
        hashTable.put("cherry", 30);
        hashTable.put("date", 40);
        hashTable.put("elderberry", 50);
        
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
        
        hashTable.display();
        
        // Check containment
        System.out.println("\nContainment checks:");
        System.out.println("Contains apple: " + hashTable.containsKey("apple"));
        System.out.println("Contains banana: " + hashTable.containsKey("banana"));
        
        // Display keys and values
        System.out.println("\nAll keys: " + hashTable.keySet());
        System.out.println("All values: " + hashTable.values());
        
        // Insert more to trigger resize
        System.out.println("\nInserting more items to trigger resize:");
        hashTable.put("fig", 60);
        hashTable.put("grape", 70);
        hashTable.put("honeydew", 80);
        
        hashTable.display();
    }
}
