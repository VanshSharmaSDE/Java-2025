package Algorithms.DataStructures.Trees;

/**
 * AVL Tree (Self-Balancing Binary Search Tree) Implementation
 * Maintains balance factor of -1, 0, or +1 for all nodes
 */
public class AVLTree<T extends Comparable<T>> {
    
    private Node<T> root;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> left, right;
        int height;
        
        Node(T data) {
            this.data = data;
            this.height = 1;
        }
    }
    
    public AVLTree() {
        this.root = null;
        this.size = 0;
    }
    
    /**
     * Insert element maintaining AVL property
     * Time Complexity: O(log n)
     */
    public void insert(T data) {
        root = insert(root, data);
        size++;
    }
    
    private Node<T> insert(Node<T> node, T data) {
        // Standard BST insertion
        if (node == null) {
            return new Node<>(data);
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = insert(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = insert(node.right, data);
        } else {
            return node; // Duplicate values not allowed
        }
        
        // Update height
        node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
        
        // Get balance factor
        int balance = getBalance(node);
        
        // Left Left Case
        if (balance > 1 && data.compareTo(node.left.data) < 0) {
            return rightRotate(node);
        }
        
        // Right Right Case
        if (balance < -1 && data.compareTo(node.right.data) > 0) {
            return leftRotate(node);
        }
        
        // Left Right Case
        if (balance > 1 && data.compareTo(node.left.data) > 0) {
            node.left = leftRotate(node.left);
            return rightRotate(node);
        }
        
        // Right Left Case
        if (balance < -1 && data.compareTo(node.right.data) < 0) {
            node.right = rightRotate(node.right);
            return leftRotate(node);
        }
        
        return node;
    }
    
    /**
     * Delete element maintaining AVL property
     * Time Complexity: O(log n)
     */
    public void delete(T data) {
        root = delete(root, data);
        size--;
    }
    
    private Node<T> delete(Node<T> node, T data) {
        if (node == null) {
            return node;
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = delete(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = delete(node.right, data);
        } else {
            // Node to be deleted found
            if (node.left == null || node.right == null) {
                Node<T> temp = (node.left != null) ? node.left : node.right;
                
                if (temp == null) {
                    temp = node;
                    node = null;
                } else {
                    node = temp;
                }
            } else {
                // Node with two children
                Node<T> temp = getMinValueNode(node.right);
                node.data = temp.data;
                node.right = delete(node.right, temp.data);
            }
        }
        
        if (node == null) {
            return node;
        }
        
        // Update height
        node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
        
        // Get balance factor
        int balance = getBalance(node);
        
        // Left Left Case
        if (balance > 1 && getBalance(node.left) >= 0) {
            return rightRotate(node);
        }
        
        // Left Right Case
        if (balance > 1 && getBalance(node.left) < 0) {
            node.left = leftRotate(node.left);
            return rightRotate(node);
        }
        
        // Right Right Case
        if (balance < -1 && getBalance(node.right) <= 0) {
            return leftRotate(node);
        }
        
        // Right Left Case
        if (balance < -1 && getBalance(node.right) > 0) {
            node.right = rightRotate(node.right);
            return leftRotate(node);
        }
        
        return node;
    }
    
    /**
     * Search for element
     * Time Complexity: O(log n)
     */
    public boolean search(T data) {
        return search(root, data);
    }
    
    private boolean search(Node<T> node, T data) {
        if (node == null) {
            return false;
        }
        
        if (data.compareTo(node.data) == 0) {
            return true;
        } else if (data.compareTo(node.data) < 0) {
            return search(node.left, data);
        } else {
            return search(node.right, data);
        }
    }
    
    // Rotation methods
    private Node<T> rightRotate(Node<T> y) {
        Node<T> x = y.left;
        Node<T> T2 = x.right;
        
        // Perform rotation
        x.right = y;
        y.left = T2;
        
        // Update heights
        y.height = Math.max(getHeight(y.left), getHeight(y.right)) + 1;
        x.height = Math.max(getHeight(x.left), getHeight(x.right)) + 1;
        
        return x;
    }
    
    private Node<T> leftRotate(Node<T> x) {
        Node<T> y = x.right;
        Node<T> T2 = y.left;
        
        // Perform rotation
        y.left = x;
        x.right = T2;
        
        // Update heights
        x.height = Math.max(getHeight(x.left), getHeight(x.right)) + 1;
        y.height = Math.max(getHeight(y.left), getHeight(y.right)) + 1;
        
        return y;
    }
    
    // Helper methods
    private int getHeight(Node<T> node) {
        return (node == null) ? 0 : node.height;
    }
    
    private int getBalance(Node<T> node) {
        return (node == null) ? 0 : getHeight(node.left) - getHeight(node.right);
    }
    
    private Node<T> getMinValueNode(Node<T> node) {
        Node<T> current = node;
        while (current.left != null) {
            current = current.left;
        }
        return current;
    }
    
    /**
     * Get all elements in order
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
     * Level order traversal
     * Time Complexity: O(n)
     */
    public java.util.List<T> levelOrderTraversal() {
        java.util.List<T> result = new java.util.ArrayList<>();
        if (root == null) return result;
        
        java.util.Queue<Node<T>> queue = new java.util.LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            Node<T> node = queue.poll();
            result.add(node.data);
            
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        
        return result;
    }
    
    /**
     * Check if tree is balanced
     * Time Complexity: O(n)
     */
    public boolean isBalanced() {
        return isBalanced(root) != -1;
    }
    
    private int isBalanced(Node<T> node) {
        if (node == null) return 0;
        
        int leftHeight = isBalanced(node.left);
        if (leftHeight == -1) return -1;
        
        int rightHeight = isBalanced(node.right);
        if (rightHeight == -1) return -1;
        
        if (Math.abs(leftHeight - rightHeight) > 1) return -1;
        
        return Math.max(leftHeight, rightHeight) + 1;
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return root == null;
    }
    
    /**
     * Display tree structure
     */
    public void display() {
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }
        
        System.out.println("AVL Tree (Level Order):");
        java.util.List<T> levelOrder = levelOrderTraversal();
        System.out.println(levelOrder);
        
        System.out.println("AVL Tree (Inorder):");
        java.util.List<T> inorder = inorderTraversal();
        System.out.println(inorder);
        
        System.out.println("Tree height: " + getHeight(root));
        System.out.println("Is balanced: " + isBalanced());
    }
    
    public static void main(String[] args) {
        System.out.println("AVL Tree Demo:");
        System.out.println("==============");
        
        AVLTree<Integer> avl = new AVLTree<>();
        
        // Insert elements
        System.out.println("Inserting: 10, 20, 30, 40, 50, 25");
        avl.insert(10);
        avl.insert(20);
        avl.insert(30);
        avl.insert(40);
        avl.insert(50);
        avl.insert(25);
        
        avl.display();
        
        // Search operations
        System.out.println("\nSearch operations:");
        System.out.println("Search 30: " + avl.search(30));
        System.out.println("Search 35: " + avl.search(35));
        
        // Delete operations
        System.out.println("\nDeleting 30:");
        avl.delete(30);
        avl.display();
        
        System.out.println("\nDeleting 40:");
        avl.delete(40);
        avl.display();
        
        // Performance test
        System.out.println("\nPerformance Test - Inserting 1000 elements:");
        AVLTree<Integer> perfTest = new AVLTree<>();
        long startTime = System.nanoTime();
        
        for (int i = 1; i <= 1000; i++) {
            perfTest.insert(i);
        }
        
        long endTime = System.nanoTime();
        System.out.println("Time taken: " + (endTime - startTime) / 1_000_000.0 + " ms");
        System.out.println("Tree height: " + perfTest.getHeight(perfTest.root));
        System.out.println("Is balanced: " + perfTest.isBalanced());
    }
}
