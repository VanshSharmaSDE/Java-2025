package Algorithms.DataStructures.Trees;

/**
 * B-Tree Implementation
 * Self-balancing tree data structure for databases and file systems
 */
public class BTree<T extends Comparable<T>> {
    
    private int minDegree; // Minimum degree (t)
    private Node<T> root;
    
    private static class Node<T> {
        int numKeys;
        T[] keys;
        Node<T>[] children;
        boolean isLeaf;
        
        @SuppressWarnings("unchecked")
        Node(int degree, boolean isLeaf) {
            this.isLeaf = isLeaf;
            this.keys = (T[]) new Comparable[2 * degree - 1];
            this.children = new Node[2 * degree];
            this.numKeys = 0;
        }
    }
    
    public BTree(int minDegree) {
        this.minDegree = minDegree;
        this.root = new Node<>(minDegree, true);
    }
    
    /**
     * Search for key in B-tree
     * Time Complexity: O(log n)
     */
    public boolean search(T key) {
        return search(root, key) != null;
    }
    
    private Node<T> search(Node<T> node, T key) {
        int i = 0;
        
        // Find first key greater than or equal to key
        while (i < node.numKeys && key.compareTo(node.keys[i]) > 0) {
            i++;
        }
        
        // If key found
        if (i < node.numKeys && key.compareTo(node.keys[i]) == 0) {
            return node;
        }
        
        // If leaf node, key not found
        if (node.isLeaf) {
            return null;
        }
        
        // Recursively search in appropriate child
        return search(node.children[i], key);
    }
    
    /**
     * Insert key into B-tree
     * Time Complexity: O(log n)
     */
    public void insert(T key) {
        Node<T> root = this.root;
        
        // If root is full, split it
        if (root.numKeys == 2 * minDegree - 1) {
            Node<T> newRoot = new Node<>(minDegree, false);
            this.root = newRoot;
            newRoot.children[0] = root;
            
            splitChild(newRoot, 0);
            insertNonFull(newRoot, key);
        } else {
            insertNonFull(root, key);
        }
    }
    
    /**
     * Insert key into non-full node
     */
    private void insertNonFull(Node<T> node, T key) {
        int i = node.numKeys - 1;
        
        if (node.isLeaf) {
            // Shift keys to make room for new key
            while (i >= 0 && key.compareTo(node.keys[i]) < 0) {
                node.keys[i + 1] = node.keys[i];
                i--;
            }
            
            node.keys[i + 1] = key;
            node.numKeys++;
        } else {
            // Find child to insert into
            while (i >= 0 && key.compareTo(node.keys[i]) < 0) {
                i--;
            }
            i++;
            
            // If child is full, split it
            if (node.children[i].numKeys == 2 * minDegree - 1) {
                splitChild(node, i);
                
                if (key.compareTo(node.keys[i]) > 0) {
                    i++;
                }
            }
            
            insertNonFull(node.children[i], key);
        }
    }
    
    /**
     * Split child y of node x at index i
     */
    private void splitChild(Node<T> x, int i) {
        Node<T> y = x.children[i];
        Node<T> z = new Node<>(minDegree, y.isLeaf);
        
        z.numKeys = minDegree - 1;
        
        // Copy latter half of keys from y to z
        for (int j = 0; j < minDegree - 1; j++) {
            z.keys[j] = y.keys[j + minDegree];
        }
        
        // Copy latter half of children from y to z
        if (!y.isLeaf) {
            for (int j = 0; j < minDegree; j++) {
                z.children[j] = y.children[j + minDegree];
            }
        }
        
        y.numKeys = minDegree - 1;
        
        // Shift children of x to make room for z
        for (int j = x.numKeys; j >= i + 1; j--) {
            x.children[j + 1] = x.children[j];
        }
        
        x.children[i + 1] = z;
        
        // Shift keys of x to make room for middle key of y
        for (int j = x.numKeys - 1; j >= i; j--) {
            x.keys[j + 1] = x.keys[j];
        }
        
        x.keys[i] = y.keys[minDegree - 1];
        x.numKeys++;
    }
    
    /**
     * Delete key from B-tree
     * Time Complexity: O(log n)
     */
    public void delete(T key) {
        delete(root, key);
        
        // If root becomes empty, make its first child the new root
        if (root.numKeys == 0) {
            if (!root.isLeaf) {
                root = root.children[0];
            }
        }
    }
    
    private void delete(Node<T> node, T key) {
        int idx = findKey(node, key);
        
        if (idx < node.numKeys && key.compareTo(node.keys[idx]) == 0) {
            // Key found in this node
            if (node.isLeaf) {
                removeFromLeaf(node, idx);
            } else {
                removeFromNonLeaf(node, idx);
            }
        } else {
            // Key not in this node
            if (node.isLeaf) {
                return; // Key not in tree
            }
            
            boolean isLastChild = (idx == node.numKeys);
            
            // If child has minimum keys, fill it
            if (node.children[idx].numKeys < minDegree) {
                fill(node, idx);
            }
            
            // After filling, the key might have moved to the previous child
            if (isLastChild && idx > node.numKeys) {
                delete(node.children[idx - 1], key);
            } else {
                delete(node.children[idx], key);
            }
        }
    }
    
    private int findKey(Node<T> node, T key) {
        int idx = 0;
        while (idx < node.numKeys && key.compareTo(node.keys[idx]) > 0) {
            idx++;
        }
        return idx;
    }
    
    private void removeFromLeaf(Node<T> node, int idx) {
        // Shift keys left
        for (int i = idx + 1; i < node.numKeys; i++) {
            node.keys[i - 1] = node.keys[i];
        }
        node.numKeys--;
    }
    
    private void removeFromNonLeaf(Node<T> node, int idx) {
        T key = node.keys[idx];
        
        // If left child has at least minDegree keys
        if (node.children[idx].numKeys >= minDegree) {
            T predecessor = getPredecessor(node, idx);
            node.keys[idx] = predecessor;
            delete(node.children[idx], predecessor);
        }
        // If right child has at least minDegree keys
        else if (node.children[idx + 1].numKeys >= minDegree) {
            T successor = getSuccessor(node, idx);
            node.keys[idx] = successor;
            delete(node.children[idx + 1], successor);
        }
        // Both children have minDegree-1 keys, merge
        else {
            merge(node, idx);
            delete(node.children[idx], key);
        }
    }
    
    private T getPredecessor(Node<T> node, int idx) {
        Node<T> current = node.children[idx];
        while (!current.isLeaf) {
            current = current.children[current.numKeys];
        }
        return current.keys[current.numKeys - 1];
    }
    
    private T getSuccessor(Node<T> node, int idx) {
        Node<T> current = node.children[idx + 1];
        while (!current.isLeaf) {
            current = current.children[0];
        }
        return current.keys[0];
    }
    
    private void fill(Node<T> node, int idx) {
        // If previous sibling has more than minDegree-1 keys, borrow from it
        if (idx != 0 && node.children[idx - 1].numKeys >= minDegree) {
            borrowFromPrev(node, idx);
        }
        // If next sibling has more than minDegree-1 keys, borrow from it
        else if (idx != node.numKeys && node.children[idx + 1].numKeys >= minDegree) {
            borrowFromNext(node, idx);
        }
        // Merge with sibling
        else {
            if (idx != node.numKeys) {
                merge(node, idx);
            } else {
                merge(node, idx - 1);
            }
        }
    }
    
    private void borrowFromPrev(Node<T> node, int idx) {
        Node<T> child = node.children[idx];
        Node<T> sibling = node.children[idx - 1];
        
        // Move key from parent to child
        for (int i = child.numKeys - 1; i >= 0; i--) {
            child.keys[i + 1] = child.keys[i];
        }
        
        if (!child.isLeaf) {
            for (int i = child.numKeys; i >= 0; i--) {
                child.children[i + 1] = child.children[i];
            }
        }
        
        child.keys[0] = node.keys[idx - 1];
        
        if (!child.isLeaf) {
            child.children[0] = sibling.children[sibling.numKeys];
        }
        
        node.keys[idx - 1] = sibling.keys[sibling.numKeys - 1];
        
        child.numKeys++;
        sibling.numKeys--;
    }
    
    private void borrowFromNext(Node<T> node, int idx) {
        Node<T> child = node.children[idx];
        Node<T> sibling = node.children[idx + 1];
        
        child.keys[child.numKeys] = node.keys[idx];
        
        if (!child.isLeaf) {
            child.children[child.numKeys + 1] = sibling.children[0];
        }
        
        node.keys[idx] = sibling.keys[0];
        
        for (int i = 1; i < sibling.numKeys; i++) {
            sibling.keys[i - 1] = sibling.keys[i];
        }
        
        if (!sibling.isLeaf) {
            for (int i = 1; i <= sibling.numKeys; i++) {
                sibling.children[i - 1] = sibling.children[i];
            }
        }
        
        child.numKeys++;
        sibling.numKeys--;
    }
    
    private void merge(Node<T> node, int idx) {
        Node<T> child = node.children[idx];
        Node<T> sibling = node.children[idx + 1];
        
        // Pull key from current node and merge with sibling
        child.keys[minDegree - 1] = node.keys[idx];
        
        for (int i = 0; i < sibling.numKeys; i++) {
            child.keys[i + minDegree] = sibling.keys[i];
        }
        
        if (!child.isLeaf) {
            for (int i = 0; i <= sibling.numKeys; i++) {
                child.children[i + minDegree] = sibling.children[i];
            }
        }
        
        // Move keys and children in current node
        for (int i = idx + 1; i < node.numKeys; i++) {
            node.keys[i - 1] = node.keys[i];
        }
        
        for (int i = idx + 2; i <= node.numKeys; i++) {
            node.children[i - 1] = node.children[i];
        }
        
        child.numKeys += sibling.numKeys + 1;
        node.numKeys--;
    }
    
    /**
     * Traverse B-tree in order
     * Time Complexity: O(n)
     */
    public java.util.List<T> inorderTraversal() {
        java.util.List<T> result = new java.util.ArrayList<>();
        inorderTraversal(root, result);
        return result;
    }
    
    private void inorderTraversal(Node<T> node, java.util.List<T> result) {
        int i;
        for (i = 0; i < node.numKeys; i++) {
            if (!node.isLeaf) {
                inorderTraversal(node.children[i], result);
            }
            result.add(node.keys[i]);
        }
        
        if (!node.isLeaf) {
            inorderTraversal(node.children[i], result);
        }
    }
    
    /**
     * Get height of B-tree
     */
    public int getHeight() {
        return getHeight(root);
    }
    
    private int getHeight(Node<T> node) {
        if (node.isLeaf) {
            return 1;
        }
        return 1 + getHeight(node.children[0]);
    }
    
    /**
     * Display B-tree structure
     */
    public void display() {
        System.out.println("B-Tree (degree " + minDegree + "):");
        display(root, 0);
        
        System.out.println("\nInorder traversal: " + inorderTraversal());
        System.out.println("Height: " + getHeight());
    }
    
    private void display(Node<T> node, int level) {
        if (node != null) {
            System.out.print("Level " + level + ": ");
            for (int i = 0; i < node.numKeys; i++) {
                System.out.print(node.keys[i] + " ");
            }
            System.out.println();
            
            if (!node.isLeaf) {
                for (int i = 0; i <= node.numKeys; i++) {
                    display(node.children[i], level + 1);
                }
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("B-Tree Demo:");
        System.out.println("============");
        
        BTree<Integer> btree = new BTree<>(3); // Minimum degree 3
        
        // Insert elements
        System.out.println("Inserting: 10, 20, 5, 6, 12, 30, 7, 17");
        int[] elements = {10, 20, 5, 6, 12, 30, 7, 17};
        for (int element : elements) {
            btree.insert(element);
        }
        
        btree.display();
        
        // Search operations
        System.out.println("\nSearch operations:");
        System.out.println("Search 6: " + btree.search(6));
        System.out.println("Search 15: " + btree.search(15));
        
        // Insert more elements to trigger splits
        System.out.println("\nInserting more elements: 25, 40, 50");
        btree.insert(25);
        btree.insert(40);
        btree.insert(50);
        
        btree.display();
        
        // Delete operations
        System.out.println("\nDeleting 6:");
        btree.delete(6);
        btree.display();
        
        System.out.println("\nDeleting 12:");
        btree.delete(12);
        btree.display();
        
        // Performance test
        System.out.println("\nPerformance Test - B-Tree with degree 50:");
        BTree<Integer> perfTest = new BTree<>(50);
        long startTime = System.nanoTime();
        
        for (int i = 1; i <= 10000; i++) {
            perfTest.insert(i);
        }
        
        long endTime = System.nanoTime();
        System.out.println("Insertion time for 10,000 elements: " + 
                          (endTime - startTime) / 1_000_000.0 + " ms");
        System.out.println("Tree height: " + perfTest.getHeight());
        
        // Test search performance
        startTime = System.nanoTime();
        boolean found = perfTest.search(5000);
        endTime = System.nanoTime();
        System.out.println("Search time: " + (endTime - startTime) / 1_000.0 + " microseconds");
        System.out.println("Element found: " + found);
    }
}
