package Algorithms.DataStructures.HashTables;

/**
 * Hash Table Algorithms and Applications
 */
public class HashTableAlgorithms {
    
    /**
     * Robin Hood Hashing - Linear probing variant
     * Minimizes variance in probe distances
     */
    public static class RobinHoodHashTable<K, V> {
        private static class Entry<K, V> {
            K key;
            V value;
            int distance; // Distance from ideal position
            
            Entry(K key, V value, int distance) {
                this.key = key;
                this.value = value;
                this.distance = distance;
            }
        }
        
        private Entry<K, V>[] table;
        private int size;
        private int capacity;
        
        @SuppressWarnings("unchecked")
        public RobinHoodHashTable(int capacity) {
            this.capacity = capacity;
            this.table = new Entry[capacity];
            this.size = 0;
        }
        
        private int hash(K key) {
            return Math.abs(key.hashCode()) % capacity;
        }
        
        public void put(K key, V value) {
            int index = hash(key);
            Entry<K, V> entry = new Entry<>(key, value, 0);
            
            while (table[index] != null) {
                // If existing key, update
                if (table[index].key.equals(key)) {
                    table[index].value = value;
                    return;
                }
                
                // Robin Hood: if current entry is richer, swap
                if (entry.distance > table[index].distance) {
                    Entry<K, V> temp = table[index];
                    table[index] = entry;
                    entry = temp;
                }
                
                index = (index + 1) % capacity;
                entry.distance++;
            }
            
            table[index] = entry;
            size++;
        }
        
        public V get(K key) {
            int index = hash(key);
            int distance = 0;
            
            while (table[index] != null) {
                if (table[index].key.equals(key)) {
                    return table[index].value;
                }
                
                // If we've gone further than this entry's distance,
                // the key doesn't exist
                if (distance > table[index].distance) {
                    break;
                }
                
                index = (index + 1) % capacity;
                distance++;
            }
            
            return null;
        }
        
        public void display() {
            System.out.println("Robin Hood Hash Table:");
            for (int i = 0; i < capacity; i++) {
                if (table[i] != null) {
                    System.out.println("Index " + i + ": [" + table[i].key + 
                                     ":" + table[i].value + "] (distance: " + 
                                     table[i].distance + ")");
                } else {
                    System.out.println("Index " + i + ": empty");
                }
            }
        }
    }
    
    /**
     * Cuckoo Hashing - Guarantees O(1) worst-case lookup
     */
    public static class CuckooHashTable<K, V> {
        private K[] keys1, keys2;
        private V[] values1, values2;
        private int size;
        private int capacity;
        private static final int MAX_ITERATIONS = 8;
        
        @SuppressWarnings("unchecked")
        public CuckooHashTable(int capacity) {
            this.capacity = capacity;
            this.keys1 = (K[]) new Object[capacity];
            this.keys2 = (K[]) new Object[capacity];
            this.values1 = (V[]) new Object[capacity];
            this.values2 = (V[]) new Object[capacity];
            this.size = 0;
        }
        
        private int hash1(K key) {
            return Math.abs(key.hashCode()) % capacity;
        }
        
        private int hash2(K key) {
            return Math.abs((key.hashCode() * 31 + 17)) % capacity;
        }
        
        public void put(K key, V value) {
            if (putHelper(key, value)) {
                size++;
            }
        }
        
        private boolean putHelper(K key, V value) {
            K currentKey = key;
            V currentValue = value;
            
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                int index1 = hash1(currentKey);
                
                // Try first table
                if (keys1[index1] == null) {
                    keys1[index1] = currentKey;
                    values1[index1] = currentValue;
                    return true;
                }
                
                // If key exists, update
                if (keys1[index1].equals(currentKey)) {
                    values1[index1] = currentValue;
                    return false;
                }
                
                // Evict from first table
                K tempKey = keys1[index1];
                V tempValue = values1[index1];
                keys1[index1] = currentKey;
                values1[index1] = currentValue;
                currentKey = tempKey;
                currentValue = tempValue;
                
                int index2 = hash2(currentKey);
                
                // Try second table
                if (keys2[index2] == null) {
                    keys2[index2] = currentKey;
                    values2[index2] = currentValue;
                    return true;
                }
                
                // If key exists, update
                if (keys2[index2].equals(currentKey)) {
                    values2[index2] = currentValue;
                    return false;
                }
                
                // Evict from second table
                tempKey = keys2[index2];
                tempValue = values2[index2];
                keys2[index2] = currentKey;
                values2[index2] = currentValue;
                currentKey = tempKey;
                currentValue = tempValue;
            }
            
            // Rehash if too many iterations
            rehash();
            return putHelper(key, value);
        }
        
        public V get(K key) {
            int index1 = hash1(key);
            if (keys1[index1] != null && keys1[index1].equals(key)) {
                return values1[index1];
            }
            
            int index2 = hash2(key);
            if (keys2[index2] != null && keys2[index2].equals(key)) {
                return values2[index2];
            }
            
            return null;
        }
        
        @SuppressWarnings("unchecked")
        private void rehash() {
            K[] oldKeys1 = keys1, oldKeys2 = keys2;
            V[] oldValues1 = values1, oldValues2 = values2;
            int oldCapacity = capacity;
            
            capacity *= 2;
            keys1 = (K[]) new Object[capacity];
            keys2 = (K[]) new Object[capacity];
            values1 = (V[]) new Object[capacity];
            values2 = (V[]) new Object[capacity];
            size = 0;
            
            // Reinsert all elements
            for (int i = 0; i < oldCapacity; i++) {
                if (oldKeys1[i] != null) {
                    put(oldKeys1[i], oldValues1[i]);
                }
                if (oldKeys2[i] != null) {
                    put(oldKeys2[i], oldValues2[i]);
                }
            }
        }
        
        public void display() {
            System.out.println("Cuckoo Hash Table:");
            System.out.println("Table 1:");
            for (int i = 0; i < capacity; i++) {
                if (keys1[i] != null) {
                    System.out.println("  Index " + i + ": [" + keys1[i] + ":" + values1[i] + "]");
                }
            }
            System.out.println("Table 2:");
            for (int i = 0; i < capacity; i++) {
                if (keys2[i] != null) {
                    System.out.println("  Index " + i + ": [" + keys2[i] + ":" + values2[i] + "]");
                }
            }
        }
    }
    
    /**
     * Hash Set implementation using chaining
     */
    public static class HashSet<T> {
        private java.util.LinkedList<T>[] buckets;
        private int size;
        private int capacity;
        
        @SuppressWarnings("unchecked")
        public HashSet(int capacity) {
            this.capacity = capacity;
            this.buckets = new java.util.LinkedList[capacity];
            this.size = 0;
            
            for (int i = 0; i < capacity; i++) {
                buckets[i] = new java.util.LinkedList<>();
            }
        }
        
        private int hash(T item) {
            return Math.abs(item.hashCode()) % capacity;
        }
        
        public boolean add(T item) {
            int index = hash(item);
            if (!buckets[index].contains(item)) {
                buckets[index].add(item);
                size++;
                return true;
            }
            return false;
        }
        
        public boolean contains(T item) {
            int index = hash(item);
            return buckets[index].contains(item);
        }
        
        public boolean remove(T item) {
            int index = hash(item);
            if (buckets[index].remove(item)) {
                size--;
                return true;
            }
            return false;
        }
        
        public int size() {
            return size;
        }
        
        public boolean isEmpty() {
            return size == 0;
        }
    }
    
    /**
     * Consistent Hashing for distributed systems
     */
    public static class ConsistentHashing {
        private java.util.TreeMap<Integer, String> ring;
        private int virtualNodes;
        
        public ConsistentHashing(int virtualNodes) {
            this.ring = new java.util.TreeMap<>();
            this.virtualNodes = virtualNodes;
        }
        
        public void addNode(String node) {
            for (int i = 0; i < virtualNodes; i++) {
                int hash = hash(node + ":" + i);
                ring.put(hash, node);
            }
        }
        
        public void removeNode(String node) {
            for (int i = 0; i < virtualNodes; i++) {
                int hash = hash(node + ":" + i);
                ring.remove(hash);
            }
        }
        
        public String getNode(String key) {
            if (ring.isEmpty()) {
                return null;
            }
            
            int hash = hash(key);
            java.util.Map.Entry<Integer, String> entry = ring.ceilingEntry(hash);
            
            if (entry == null) {
                entry = ring.firstEntry();
            }
            
            return entry.getValue();
        }
        
        private int hash(String key) {
            return Math.abs(key.hashCode());
        }
        
        public void displayRing() {
            System.out.println("Consistent Hash Ring:");
            for (java.util.Map.Entry<Integer, String> entry : ring.entrySet()) {
                System.out.println("Hash: " + entry.getKey() + " -> Node: " + entry.getValue());
            }
        }
    }
    
    /**
     * Bloom Filter - Probabilistic data structure
     */
    public static class BloomFilter {
        private boolean[] bitArray;
        private int size;
        private int hashFunctions;
        
        public BloomFilter(int size, int hashFunctions) {
            this.size = size;
            this.hashFunctions = hashFunctions;
            this.bitArray = new boolean[size];
        }
        
        public void add(String item) {
            for (int i = 0; i < hashFunctions; i++) {
                int hash = hash(item, i) % size;
                bitArray[hash] = true;
            }
        }
        
        public boolean mightContain(String item) {
            for (int i = 0; i < hashFunctions; i++) {
                int hash = hash(item, i) % size;
                if (!bitArray[hash]) {
                    return false;
                }
            }
            return true;
        }
        
        private int hash(String item, int seed) {
            return Math.abs((item.hashCode() * 31 + seed));
        }
        
        public double getFalsePositiveRate() {
            int setBits = 0;
            for (boolean bit : bitArray) {
                if (bit) setBits++;
            }
            
            double ratio = (double) setBits / size;
            return Math.pow(ratio, hashFunctions);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Hash Table Algorithms Demo:");
        System.out.println("===========================");
        
        // Robin Hood Hashing
        System.out.println("1. Robin Hood Hashing:");
        RobinHoodHashTable<String, Integer> robinHood = new RobinHoodHashTable<>(8);
        robinHood.put("apple", 10);
        robinHood.put("banana", 20);
        robinHood.put("cherry", 30);
        robinHood.display();
        
        System.out.println("Get apple: " + robinHood.get("apple"));
        
        // Cuckoo Hashing
        System.out.println("\n2. Cuckoo Hashing:");
        CuckooHashTable<String, Integer> cuckoo = new CuckooHashTable<>(4);
        cuckoo.put("key1", 100);
        cuckoo.put("key2", 200);
        cuckoo.put("key3", 300);
        cuckoo.display();
        
        System.out.println("Get key2: " + cuckoo.get("key2"));
        
        // Hash Set
        System.out.println("\n3. Hash Set:");
        HashSet<String> hashSet = new HashSet<>(10);
        hashSet.add("item1");
        hashSet.add("item2");
        hashSet.add("item3");
        hashSet.add("item1"); // Duplicate
        
        System.out.println("Size: " + hashSet.size());
        System.out.println("Contains item1: " + hashSet.contains("item1"));
        System.out.println("Contains item4: " + hashSet.contains("item4"));
        
        // Consistent Hashing
        System.out.println("\n4. Consistent Hashing:");
        ConsistentHashing ch = new ConsistentHashing(3);
        ch.addNode("Server1");
        ch.addNode("Server2");
        ch.addNode("Server3");
        
        System.out.println("Key 'user123' maps to: " + ch.getNode("user123"));
        System.out.println("Key 'data456' maps to: " + ch.getNode("data456"));
        
        // Bloom Filter
        System.out.println("\n5. Bloom Filter:");
        BloomFilter bloomFilter = new BloomFilter(1000, 3);
        bloomFilter.add("apple");
        bloomFilter.add("banana");
        bloomFilter.add("cherry");
        
        System.out.println("Might contain apple: " + bloomFilter.mightContain("apple"));
        System.out.println("Might contain grape: " + bloomFilter.mightContain("grape"));
        System.out.println("False positive rate: " + 
                          String.format("%.4f", bloomFilter.getFalsePositiveRate()));
    }
}
