package Algorithms.DataStructures.Trees;

import java.util.*;

/**
 * Binary Search Tree Implementation with various algorithms
 */
public class BinarySearchTree {
    
    /**
     * TreeNode class representing each node in the BST
     */
    static class TreeNode {
        int data;
        TreeNode left, right;
        
        public TreeNode(int data) {
            this.data = data;
            this.left = this.right = null;
        }
    }
    
    private TreeNode root;
    
    public BinarySearchTree() {
        root = null;
    }
    
    /**
     * Insert a value into the BST
     * @param data Value to insert
     */
    public void insert(int data) {
        root = insertRec(root, data);
    }
    
    private TreeNode insertRec(TreeNode root, int data) {
        if (root == null) {
            root = new TreeNode(data);
            return root;
        }
        
        if (data < root.data) {
            root.left = insertRec(root.left, data);
        } else if (data > root.data) {
            root.right = insertRec(root.right, data);
        }
        
        return root;
    }
    
    /**
     * Search for a value in the BST
     * @param data Value to search
     * @return true if found, false otherwise
     */
    public boolean search(int data) {
        return searchRec(root, data);
    }
    
    private boolean searchRec(TreeNode root, int data) {
        if (root == null) {
            return false;
        }
        
        if (data == root.data) {
            return true;
        }
        
        return data < root.data ? searchRec(root.left, data) : searchRec(root.right, data);
    }
    
    /**
     * Delete a value from the BST
     * @param data Value to delete
     */
    public void delete(int data) {
        root = deleteRec(root, data);
    }
    
    private TreeNode deleteRec(TreeNode root, int data) {
        if (root == null) {
            return root;
        }
        
        if (data < root.data) {
            root.left = deleteRec(root.left, data);
        } else if (data > root.data) {
            root.right = deleteRec(root.right, data);
        } else {
            // Node to be deleted found
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            }
            
            // Node with two children
            root.data = minValue(root.right);
            root.right = deleteRec(root.right, root.data);
        }
        
        return root;
    }
    
    private int minValue(TreeNode root) {
        int minv = root.data;
        while (root.left != null) {
            minv = root.left.data;
            root = root.left;
        }
        return minv;
    }
    
    /**
     * Inorder traversal (Left, Root, Right)
     */
    public void inorderTraversal() {
        System.out.print("Inorder: ");
        inorderRec(root);
        System.out.println();
    }
    
    private void inorderRec(TreeNode root) {
        if (root != null) {
            inorderRec(root.left);
            System.out.print(root.data + " ");
            inorderRec(root.right);
        }
    }
    
    /**
     * Preorder traversal (Root, Left, Right)
     */
    public void preorderTraversal() {
        System.out.print("Preorder: ");
        preorderRec(root);
        System.out.println();
    }
    
    private void preorderRec(TreeNode root) {
        if (root != null) {
            System.out.print(root.data + " ");
            preorderRec(root.left);
            preorderRec(root.right);
        }
    }
    
    /**
     * Postorder traversal (Left, Right, Root)
     */
    public void postorderTraversal() {
        System.out.print("Postorder: ");
        postorderRec(root);
        System.out.println();
    }
    
    private void postorderRec(TreeNode root) {
        if (root != null) {
            postorderRec(root.left);
            postorderRec(root.right);
            System.out.print(root.data + " ");
        }
    }
    
    /**
     * Level order traversal (BFS)
     */
    public void levelOrderTraversal() {
        if (root == null) return;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        System.out.print("Level order: ");
        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            System.out.print(current.data + " ");
            
            if (current.left != null) queue.add(current.left);
            if (current.right != null) queue.add(current.right);
        }
        System.out.println();
    }
    
    /**
     * Calculate height of the tree
     * @return Height of tree
     */
    public int height() {
        return heightRec(root);
    }
    
    private int heightRec(TreeNode root) {
        if (root == null) {
            return -1; // Height of empty tree is -1
        }
        
        return 1 + Math.max(heightRec(root.left), heightRec(root.right));
    }
    
    /**
     * Count total number of nodes
     * @return Number of nodes
     */
    public int countNodes() {
        return countNodesRec(root);
    }
    
    private int countNodesRec(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        return 1 + countNodesRec(root.left) + countNodesRec(root.right);
    }
    
    /**
     * Find minimum value in the tree
     * @return Minimum value
     */
    public int findMin() {
        if (root == null) {
            throw new RuntimeException("Tree is empty");
        }
        
        TreeNode current = root;
        while (current.left != null) {
            current = current.left;
        }
        return current.data;
    }
    
    /**
     * Find maximum value in the tree
     * @return Maximum value
     */
    public int findMax() {
        if (root == null) {
            throw new RuntimeException("Tree is empty");
        }
        
        TreeNode current = root;
        while (current.right != null) {
            current = current.right;
        }
        return current.data;
    }
    
    /**
     * Check if tree is a valid BST
     * @return true if valid BST, false otherwise
     */
    public boolean isValidBST() {
        return isValidBSTRec(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    
    private boolean isValidBSTRec(TreeNode root, int min, int max) {
        if (root == null) {
            return true;
        }
        
        if (root.data <= min || root.data >= max) {
            return false;
        }
        
        return isValidBSTRec(root.left, min, root.data) && 
               isValidBSTRec(root.right, root.data, max);
    }
    
    /**
     * Find kth smallest element
     * @param k Position (1-indexed)
     * @return kth smallest element
     */
    public int kthSmallest(int k) {
        List<Integer> inorderList = new ArrayList<>();
        inorderForKth(root, inorderList);
        
        if (k <= 0 || k > inorderList.size()) {
            throw new IllegalArgumentException("Invalid k value");
        }
        
        return inorderList.get(k - 1);
    }
    
    private void inorderForKth(TreeNode root, List<Integer> list) {
        if (root != null) {
            inorderForKth(root.left, list);
            list.add(root.data);
            inorderForKth(root.right, list);
        }
    }
    
    /**
     * Find Lowest Common Ancestor of two nodes
     * @param p First node value
     * @param q Second node value
     * @return LCA node value
     */
    public int lowestCommonAncestor(int p, int q) {
        TreeNode lca = lcaRec(root, p, q);
        return lca != null ? lca.data : -1;
    }
    
    private TreeNode lcaRec(TreeNode root, int p, int q) {
        if (root == null) {
            return null;
        }
        
        if (root.data > p && root.data > q) {
            return lcaRec(root.left, p, q);
        }
        
        if (root.data < p && root.data < q) {
            return lcaRec(root.right, p, q);
        }
        
        return root;
    }
    
    public static void main(String[] args) {
        System.out.println("Binary Search Tree Algorithms:");
        System.out.println("==============================");
        
        BinarySearchTree bst = new BinarySearchTree();
        
        // Insert values
        int[] values = {50, 30, 20, 40, 70, 60, 80};
        System.out.println("Inserting values: " + Arrays.toString(values));
        
        for (int value : values) {
            bst.insert(value);
        }
        
        // Traversals
        System.out.println("\nTree Traversals:");
        bst.inorderTraversal();
        bst.preorderTraversal();
        bst.postorderTraversal();
        bst.levelOrderTraversal();
        
        // Tree properties
        System.out.println("\nTree Properties:");
        System.out.println("Height: " + bst.height());
        System.out.println("Number of nodes: " + bst.countNodes());
        System.out.println("Minimum value: " + bst.findMin());
        System.out.println("Maximum value: " + bst.findMax());
        System.out.println("Is valid BST: " + bst.isValidBST());
        
        // Search operations
        System.out.println("\nSearch Operations:");
        System.out.println("Search 40: " + bst.search(40));
        System.out.println("Search 100: " + bst.search(100));
        
        // Kth smallest
        System.out.println("3rd smallest: " + bst.kthSmallest(3));
        
        // Lowest Common Ancestor
        System.out.println("LCA of 20 and 40: " + bst.lowestCommonAncestor(20, 40));
        System.out.println("LCA of 60 and 80: " + bst.lowestCommonAncestor(60, 80));
        
        // Delete operation
        System.out.println("\nAfter deleting 30:");
        bst.delete(30);
        bst.inorderTraversal();
    }
}
