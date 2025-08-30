package Algorithms.DataStructures.Trees;

/**
 * Red-Black Tree Implementation
 * Self-balancing binary search tree with colored nodes
 */
public class RedBlackTree<T extends Comparable<T>> {
    
    private static final boolean RED = true;
    private static final boolean BLACK = false;
    
    private Node<T> root;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> left, right, parent;
        boolean color;
        
        Node(T data) {
            this.data = data;
            this.color = RED; // New nodes are always red
        }
        
        Node<T> grandparent() {
            return (parent != null) ? parent.parent : null;
        }
        
        Node<T> uncle() {
            Node<T> grandparent = grandparent();
            if (grandparent == null) return null;
            return (parent == grandparent.left) ? grandparent.right : grandparent.left;
        }
        
        Node<T> sibling() {
            if (parent == null) return null;
            return (this == parent.left) ? parent.right : parent.left;
        }
    }
    
    public RedBlackTree() {
        this.root = null;
        this.size = 0;
    }
    
    /**
     * Insert element maintaining Red-Black properties
     * Time Complexity: O(log n)
     */
    public void insert(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (root == null) {
            root = newNode;
            root.color = BLACK;
            size++;
            return;
        }
        
        // Standard BST insertion
        Node<T> parent = null;
        Node<T> current = root;
        
        while (current != null) {
            parent = current;
            if (data.compareTo(current.data) < 0) {
                current = current.left;
            } else if (data.compareTo(current.data) > 0) {
                current = current.right;
            } else {
                return; // Duplicate not allowed
            }
        }
        
        newNode.parent = parent;
        if (data.compareTo(parent.data) < 0) {
            parent.left = newNode;
        } else {
            parent.right = newNode;
        }
        
        size++;
        fixInsertViolation(newNode);
    }
    
    /**
     * Fix Red-Black tree violations after insertion
     */
    private void fixInsertViolation(Node<T> node) {
        while (node != root && node.parent.color == RED) {
            if (node.parent == node.grandparent().left) {
                Node<T> uncle = node.uncle();
                
                if (uncle != null && uncle.color == RED) {
                    // Case 1: Uncle is red
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    node.grandparent().color = RED;
                    node = node.grandparent();
                } else {
                    if (node == node.parent.right) {
                        // Case 2: Uncle is black, node is right child
                        node = node.parent;
                        leftRotate(node);
                    }
                    // Case 3: Uncle is black, node is left child
                    node.parent.color = BLACK;
                    node.grandparent().color = RED;
                    rightRotate(node.grandparent());
                }
            } else {
                Node<T> uncle = node.uncle();
                
                if (uncle != null && uncle.color == RED) {
                    // Case 1: Uncle is red
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    node.grandparent().color = RED;
                    node = node.grandparent();
                } else {
                    if (node == node.parent.left) {
                        // Case 2: Uncle is black, node is left child
                        node = node.parent;
                        rightRotate(node);
                    }
                    // Case 3: Uncle is black, node is right child
                    node.parent.color = BLACK;
                    node.grandparent().color = RED;
                    leftRotate(node.grandparent());
                }
            }
        }
        root.color = BLACK;
    }
    
    /**
     * Delete element maintaining Red-Black properties
     * Time Complexity: O(log n)
     */
    public void delete(T data) {
        Node<T> nodeToDelete = search(root, data);
        if (nodeToDelete == null) return;
        
        size--;
        deleteNode(nodeToDelete);
    }
    
    private void deleteNode(Node<T> node) {
        Node<T> replacement;
        boolean originalColor = node.color;
        
        if (node.left == null) {
            replacement = node.right;
            transplant(node, node.right);
        } else if (node.right == null) {
            replacement = node.left;
            transplant(node, node.left);
        } else {
            Node<T> successor = minimum(node.right);
            originalColor = successor.color;
            replacement = successor.right;
            
            if (successor.parent == node) {
                if (replacement != null) replacement.parent = successor;
            } else {
                transplant(successor, successor.right);
                successor.right = node.right;
                successor.right.parent = successor;
            }
            
            transplant(node, successor);
            successor.left = node.left;
            successor.left.parent = successor;
            successor.color = node.color;
        }
        
        if (originalColor == BLACK && replacement != null) {
            fixDeleteViolation(replacement);
        }
    }
    
    /**
     * Fix Red-Black tree violations after deletion
     */
    private void fixDeleteViolation(Node<T> node) {
        while (node != root && node.color == BLACK) {
            if (node == node.parent.left) {
                Node<T> sibling = node.sibling();
                
                if (sibling.color == RED) {
                    sibling.color = BLACK;
                    node.parent.color = RED;
                    leftRotate(node.parent);
                    sibling = node.parent.right;
                }
                
                if ((sibling.left == null || sibling.left.color == BLACK) &&
                    (sibling.right == null || sibling.right.color == BLACK)) {
                    sibling.color = RED;
                    node = node.parent;
                } else {
                    if (sibling.right == null || sibling.right.color == BLACK) {
                        if (sibling.left != null) sibling.left.color = BLACK;
                        sibling.color = RED;
                        rightRotate(sibling);
                        sibling = node.parent.right;
                    }
                    
                    sibling.color = node.parent.color;
                    node.parent.color = BLACK;
                    if (sibling.right != null) sibling.right.color = BLACK;
                    leftRotate(node.parent);
                    node = root;
                }
            } else {
                Node<T> sibling = node.sibling();
                
                if (sibling.color == RED) {
                    sibling.color = BLACK;
                    node.parent.color = RED;
                    rightRotate(node.parent);
                    sibling = node.parent.left;
                }
                
                if ((sibling.right == null || sibling.right.color == BLACK) &&
                    (sibling.left == null || sibling.left.color == BLACK)) {
                    sibling.color = RED;
                    node = node.parent;
                } else {
                    if (sibling.left == null || sibling.left.color == BLACK) {
                        if (sibling.right != null) sibling.right.color = BLACK;
                        sibling.color = RED;
                        leftRotate(sibling);
                        sibling = node.parent.left;
                    }
                    
                    sibling.color = node.parent.color;
                    node.parent.color = BLACK;
                    if (sibling.left != null) sibling.left.color = BLACK;
                    rightRotate(node.parent);
                    node = root;
                }
            }
        }
        node.color = BLACK;
    }
    
    // Rotation methods
    private void leftRotate(Node<T> x) {
        Node<T> y = x.right;
        x.right = y.left;
        
        if (y.left != null) {
            y.left.parent = x;
        }
        
        y.parent = x.parent;
        
        if (x.parent == null) {
            root = y;
        } else if (x == x.parent.left) {
            x.parent.left = y;
        } else {
            x.parent.right = y;
        }
        
        y.left = x;
        x.parent = y;
    }
    
    private void rightRotate(Node<T> y) {
        Node<T> x = y.left;
        y.left = x.right;
        
        if (x.right != null) {
            x.right.parent = y;
        }
        
        x.parent = y.parent;
        
        if (y.parent == null) {
            root = x;
        } else if (y == y.parent.right) {
            y.parent.right = x;
        } else {
            y.parent.left = x;
        }
        
        x.right = y;
        y.parent = x;
    }
    
    // Helper methods
    private void transplant(Node<T> u, Node<T> v) {
        if (u.parent == null) {
            root = v;
        } else if (u == u.parent.left) {
            u.parent.left = v;
        } else {
            u.parent.right = v;
        }
        
        if (v != null) {
            v.parent = u.parent;
        }
    }
    
    private Node<T> minimum(Node<T> node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }
    
    /**
     * Search for element
     * Time Complexity: O(log n)
     */
    public boolean contains(T data) {
        return search(root, data) != null;
    }
    
    private Node<T> search(Node<T> node, T data) {
        if (node == null || data.compareTo(node.data) == 0) {
            return node;
        }
        
        if (data.compareTo(node.data) < 0) {
            return search(node.left, data);
        } else {
            return search(node.right, data);
        }
    }
    
    /**
     * Inorder traversal
     * Time Complexity: O(n)
     */
    public java.util.List<T> inorderTraversal() {
        java.util.List<T> result = new java.util.ArrayList<>();
        inorderTraversal(root, result);
        return result;
    }
    
    private void inorderTraversal(Node<T> node, java.util.List<T> result) {
        if (node != null) {
            inorderTraversal(node.left, result);
            result.add(node.data);
            inorderTraversal(node.right, result);
        }
    }
    
    /**
     * Validate Red-Black tree properties
     */
    public boolean isValidRedBlackTree() {
        if (root != null && root.color != BLACK) return false;
        return isValidRedBlackTree(root) != -1;
    }
    
    private int isValidRedBlackTree(Node<T> node) {
        if (node == null) return 1;
        
        // Check red node property
        if (node.color == RED) {
            if ((node.left != null && node.left.color == RED) ||
                (node.right != null && node.right.color == RED)) {
                return -1;
            }
        }
        
        int leftBlackHeight = isValidRedBlackTree(node.left);
        int rightBlackHeight = isValidRedBlackTree(node.right);
        
        if (leftBlackHeight == -1 || rightBlackHeight == -1 || 
            leftBlackHeight != rightBlackHeight) {
            return -1;
        }
        
        return leftBlackHeight + (node.color == BLACK ? 1 : 0);
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return root == null;
    }
    
    /**
     * Display tree with colors
     */
    public void display() {
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }
        
        System.out.println("Red-Black Tree (Inorder with colors):");
        displayWithColors(root);
        System.out.println();
        
        System.out.println("Elements in order: " + inorderTraversal());
        System.out.println("Is valid RB tree: " + isValidRedBlackTree());
        System.out.println("Size: " + size);
    }
    
    private void displayWithColors(Node<T> node) {
        if (node != null) {
            displayWithColors(node.left);
            System.out.print(node.data + "(" + (node.color ? "R" : "B") + ") ");
            displayWithColors(node.right);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Red-Black Tree Demo:");
        System.out.println("====================");
        
        RedBlackTree<Integer> rbt = new RedBlackTree<>();
        
        // Insert elements
        System.out.println("Inserting: 10, 20, 30, 15, 25, 5, 1");
        rbt.insert(10);
        rbt.insert(20);
        rbt.insert(30);
        rbt.insert(15);
        rbt.insert(25);
        rbt.insert(5);
        rbt.insert(1);
        
        rbt.display();
        
        // Search operations
        System.out.println("\nSearch operations:");
        System.out.println("Contains 15: " + rbt.contains(15));
        System.out.println("Contains 35: " + rbt.contains(35));
        
        // Delete operations
        System.out.println("\nDeleting 20:");
        rbt.delete(20);
        rbt.display();
        
        System.out.println("\nDeleting 30:");
        rbt.delete(30);
        rbt.display();
        
        // Performance test
        System.out.println("\nPerformance Test - Inserting 1000 elements:");
        RedBlackTree<Integer> perfTest = new RedBlackTree<>();
        long startTime = System.nanoTime();
        
        for (int i = 1; i <= 1000; i++) {
            perfTest.insert(i);
        }
        
        long endTime = System.nanoTime();
        System.out.println("Time taken: " + (endTime - startTime) / 1_000_000.0 + " ms");
        System.out.println("Is valid RB tree: " + perfTest.isValidRedBlackTree());
        System.out.println("Size: " + perfTest.size());
    }
}
