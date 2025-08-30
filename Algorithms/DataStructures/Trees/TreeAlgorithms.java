package Algorithms.DataStructures.Trees;

/**
 * Comprehensive Tree Algorithms Collection
 * Advanced tree operations and algorithms
 */
public class TreeAlgorithms {
    
    /**
     * Generic tree node for various algorithms
     */
    public static class TreeNode<T> {
        public T data;
        public TreeNode<T> left, right;
        
        public TreeNode(T data) {
            this.data = data;
        }
    }
    
    /**
     * Tree traversal algorithms
     */
    public static class TreeTraversals {
        
        /**
         * Morris Inorder Traversal (O(1) space)
         * Time Complexity: O(n), Space: O(1)
         */
        public static <T> java.util.List<T> morrisInorder(TreeNode<T> root) {
            java.util.List<T> result = new java.util.ArrayList<>();
            TreeNode<T> current = root;
            
            while (current != null) {
                if (current.left == null) {
                    result.add(current.data);
                    current = current.right;
                } else {
                    // Find inorder predecessor
                    TreeNode<T> predecessor = current.left;
                    while (predecessor.right != null && predecessor.right != current) {
                        predecessor = predecessor.right;
                    }
                    
                    if (predecessor.right == null) {
                        predecessor.right = current;
                        current = current.left;
                    } else {
                        predecessor.right = null;
                        result.add(current.data);
                        current = current.right;
                    }
                }
            }
            
            return result;
        }
        
        /**
         * Zigzag level order traversal
         */
        public static <T> java.util.List<java.util.List<T>> zigzagLevelOrder(TreeNode<T> root) {
            java.util.List<java.util.List<T>> result = new java.util.ArrayList<>();
            if (root == null) return result;
            
            java.util.Queue<TreeNode<T>> queue = new java.util.LinkedList<>();
            queue.offer(root);
            boolean leftToRight = true;
            
            while (!queue.isEmpty()) {
                int size = queue.size();
                java.util.List<T> level = new java.util.ArrayList<>();
                
                for (int i = 0; i < size; i++) {
                    TreeNode<T> node = queue.poll();
                    
                    if (leftToRight) {
                        level.add(node.data);
                    } else {
                        level.add(0, node.data);
                    }
                    
                    if (node.left != null) queue.offer(node.left);
                    if (node.right != null) queue.offer(node.right);
                }
                
                result.add(level);
                leftToRight = !leftToRight;
            }
            
            return result;
        }
        
        /**
         * Vertical order traversal
         */
        public static <T> java.util.List<java.util.List<T>> verticalTraversal(TreeNode<T> root) {
            java.util.List<java.util.List<T>> result = new java.util.ArrayList<>();
            if (root == null) return result;
            
            java.util.Map<Integer, java.util.List<T>> map = new java.util.TreeMap<>();
            java.util.Queue<TreeNode<T>> nodeQueue = new java.util.LinkedList<>();
            java.util.Queue<Integer> colQueue = new java.util.LinkedList<>();
            
            nodeQueue.offer(root);
            colQueue.offer(0);
            
            while (!nodeQueue.isEmpty()) {
                TreeNode<T> node = nodeQueue.poll();
                int col = colQueue.poll();
                
                map.computeIfAbsent(col, k -> new java.util.ArrayList<>()).add(node.data);
                
                if (node.left != null) {
                    nodeQueue.offer(node.left);
                    colQueue.offer(col - 1);
                }
                
                if (node.right != null) {
                    nodeQueue.offer(node.right);
                    colQueue.offer(col + 1);
                }
            }
            
            for (java.util.List<T> values : map.values()) {
                result.add(values);
            }
            
            return result;
        }
    }
    
    /**
     * Tree construction algorithms
     */
    public static class TreeConstruction {
        
        /**
         * Build tree from inorder and preorder traversals
         */
        public static TreeNode<Integer> buildTreeInorderPreorder(int[] inorder, int[] preorder) {
            java.util.Map<Integer, Integer> inorderMap = new java.util.HashMap<>();
            for (int i = 0; i < inorder.length; i++) {
                inorderMap.put(inorder[i], i);
            }
            
            return buildTree(preorder, 0, 0, inorder.length - 1, inorderMap);
        }
        
        private static TreeNode<Integer> buildTree(int[] preorder, int preStart, 
                                                 int inStart, int inEnd, 
                                                 java.util.Map<Integer, Integer> inorderMap) {
            if (preStart >= preorder.length || inStart > inEnd) {
                return null;
            }
            
            TreeNode<Integer> root = new TreeNode<>(preorder[preStart]);
            int inIndex = inorderMap.get(preorder[preStart]);
            
            root.left = buildTree(preorder, preStart + 1, inStart, inIndex - 1, inorderMap);
            root.right = buildTree(preorder, preStart + inIndex - inStart + 1, 
                                 inIndex + 1, inEnd, inorderMap);
            
            return root;
        }
        
        /**
         * Serialize and deserialize binary tree
         */
        public static String serialize(TreeNode<Integer> root) {
            if (root == null) return "null";
            return root.data + "," + serialize(root.left) + "," + serialize(root.right);
        }
        
        public static TreeNode<Integer> deserialize(String data) {
            java.util.Queue<String> queue = new java.util.LinkedList<>();
            java.util.Collections.addAll(queue, data.split(","));
            return deserializeHelper(queue);
        }
        
        private static TreeNode<Integer> deserializeHelper(java.util.Queue<String> queue) {
            String val = queue.poll();
            if ("null".equals(val)) return null;
            
            TreeNode<Integer> node = new TreeNode<>(Integer.parseInt(val));
            node.left = deserializeHelper(queue);
            node.right = deserializeHelper(queue);
            return node;
        }
    }
    
    /**
     * Tree analysis algorithms
     */
    public static class TreeAnalysis {
        
        /**
         * Find lowest common ancestor
         */
        public static <T> TreeNode<T> lowestCommonAncestor(TreeNode<T> root, 
                                                          TreeNode<T> p, TreeNode<T> q) {
            if (root == null || root == p || root == q) {
                return root;
            }
            
            TreeNode<T> left = lowestCommonAncestor(root.left, p, q);
            TreeNode<T> right = lowestCommonAncestor(root.right, p, q);
            
            if (left != null && right != null) {
                return root;
            }
            
            return left != null ? left : right;
        }
        
        /**
         * Check if tree is symmetric
         */
        public static <T> boolean isSymmetric(TreeNode<T> root) {
            return root == null || isSymmetric(root.left, root.right);
        }
        
        private static <T> boolean isSymmetric(TreeNode<T> left, TreeNode<T> right) {
            if (left == null && right == null) return true;
            if (left == null || right == null) return false;
            
            return left.data.equals(right.data) && 
                   isSymmetric(left.left, right.right) && 
                   isSymmetric(left.right, right.left);
        }
        
        /**
         * Find diameter of tree
         */
        public static <T> int diameter(TreeNode<T> root) {
            int[] maxDiameter = {0};
            height(root, maxDiameter);
            return maxDiameter[0];
        }
        
        private static <T> int height(TreeNode<T> node, int[] maxDiameter) {
            if (node == null) return 0;
            
            int left = height(node.left, maxDiameter);
            int right = height(node.right, maxDiameter);
            
            maxDiameter[0] = Math.max(maxDiameter[0], left + right);
            
            return Math.max(left, right) + 1;
        }
        
        /**
         * Check if tree is balanced
         */
        public static <T> boolean isBalanced(TreeNode<T> root) {
            return checkBalance(root) != -1;
        }
        
        private static <T> int checkBalance(TreeNode<T> node) {
            if (node == null) return 0;
            
            int left = checkBalance(node.left);
            if (left == -1) return -1;
            
            int right = checkBalance(node.right);
            if (right == -1) return -1;
            
            if (Math.abs(left - right) > 1) return -1;
            
            return Math.max(left, right) + 1;
        }
        
        /**
         * Find maximum path sum
         */
        public static int maxPathSum(TreeNode<Integer> root) {
            int[] maxSum = {Integer.MIN_VALUE};
            maxPathSumHelper(root, maxSum);
            return maxSum[0];
        }
        
        private static int maxPathSumHelper(TreeNode<Integer> node, int[] maxSum) {
            if (node == null) return 0;
            
            int left = Math.max(0, maxPathSumHelper(node.left, maxSum));
            int right = Math.max(0, maxPathSumHelper(node.right, maxSum));
            
            maxSum[0] = Math.max(maxSum[0], node.data + left + right);
            
            return node.data + Math.max(left, right);
        }
    }
    
    /**
     * Tree transformation algorithms
     */
    public static class TreeTransformation {
        
        /**
         * Flatten tree to linked list
         */
        public static <T> void flatten(TreeNode<T> root) {
            if (root == null) return;
            
            flatten(root.left);
            flatten(root.right);
            
            TreeNode<T> temp = root.right;
            root.right = root.left;
            root.left = null;
            
            while (root.right != null) {
                root = root.right;
            }
            root.right = temp;
        }
        
        /**
         * Convert sorted array to BST
         */
        public static TreeNode<Integer> sortedArrayToBST(int[] nums) {
            return sortedArrayToBST(nums, 0, nums.length - 1);
        }
        
        private static TreeNode<Integer> sortedArrayToBST(int[] nums, int left, int right) {
            if (left > right) return null;
            
            int mid = left + (right - left) / 2;
            TreeNode<Integer> root = new TreeNode<>(nums[mid]);
            
            root.left = sortedArrayToBST(nums, left, mid - 1);
            root.right = sortedArrayToBST(nums, mid + 1, right);
            
            return root;
        }
        
        /**
         * Invert binary tree
         */
        public static <T> TreeNode<T> invertTree(TreeNode<T> root) {
            if (root == null) return null;
            
            TreeNode<T> temp = root.left;
            root.left = invertTree(root.right);
            root.right = invertTree(temp);
            
            return root;
        }
    }
    
    /**
     * N-ary tree algorithms
     */
    public static class NaryTree {
        
        public static class Node<T> {
            public T data;
            public java.util.List<Node<T>> children;
            
            public Node(T data) {
                this.data = data;
                this.children = new java.util.ArrayList<>();
            }
        }
        
        /**
         * N-ary tree preorder traversal
         */
        public static <T> java.util.List<T> preorder(Node<T> root) {
            java.util.List<T> result = new java.util.ArrayList<>();
            if (root == null) return result;
            
            java.util.Stack<Node<T>> stack = new java.util.Stack<>();
            stack.push(root);
            
            while (!stack.isEmpty()) {
                Node<T> node = stack.pop();
                result.add(node.data);
                
                for (int i = node.children.size() - 1; i >= 0; i--) {
                    stack.push(node.children.get(i));
                }
            }
            
            return result;
        }
        
        /**
         * N-ary tree level order traversal
         */
        public static <T> java.util.List<java.util.List<T>> levelOrder(Node<T> root) {
            java.util.List<java.util.List<T>> result = new java.util.ArrayList<>();
            if (root == null) return result;
            
            java.util.Queue<Node<T>> queue = new java.util.LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                int size = queue.size();
                java.util.List<T> level = new java.util.ArrayList<>();
                
                for (int i = 0; i < size; i++) {
                    Node<T> node = queue.poll();
                    level.add(node.data);
                    
                    for (Node<T> child : node.children) {
                        queue.offer(child);
                    }
                }
                
                result.add(level);
            }
            
            return result;
        }
        
        /**
         * N-ary tree maximum depth
         */
        public static <T> int maxDepth(Node<T> root) {
            if (root == null) return 0;
            
            int maxChildDepth = 0;
            for (Node<T> child : root.children) {
                maxChildDepth = Math.max(maxChildDepth, maxDepth(child));
            }
            
            return maxChildDepth + 1;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Tree Algorithms Demo:");
        System.out.println("=====================");
        
        // Create sample tree:    1
        //                       / \
        //                      2   3
        //                     / \
        //                    4   5
        TreeNode<Integer> root = new TreeNode<>(1);
        root.left = new TreeNode<>(2);
        root.right = new TreeNode<>(3);
        root.left.left = new TreeNode<>(4);
        root.left.right = new TreeNode<>(5);
        
        // Morris Inorder Traversal
        System.out.println("Morris Inorder: " + TreeTraversals.morrisInorder(root));
        
        // Zigzag traversal
        System.out.println("Zigzag Level Order: " + TreeTraversals.zigzagLevelOrder(root));
        
        // Vertical traversal
        System.out.println("Vertical Traversal: " + TreeTraversals.verticalTraversal(root));
        
        // Tree analysis
        System.out.println("Tree diameter: " + TreeAnalysis.diameter(root));
        System.out.println("Is balanced: " + TreeAnalysis.isBalanced(root));
        System.out.println("Max path sum: " + TreeAnalysis.maxPathSum(root));
        
        // Serialization
        String serialized = TreeConstruction.serialize(root);
        System.out.println("Serialized: " + serialized);
        
        TreeNode<Integer> deserialized = TreeConstruction.deserialize(serialized);
        System.out.println("Deserialized inorder: " + TreeTraversals.morrisInorder(deserialized));
        
        // Array to BST
        int[] sortedArray = {1, 2, 3, 4, 5, 6, 7};
        TreeNode<Integer> bst = TreeTransformation.sortedArrayToBST(sortedArray);
        System.out.println("BST from sorted array: " + TreeTraversals.morrisInorder(bst));
        
        // N-ary tree demo
        NaryTree.Node<Integer> naryRoot = new NaryTree.Node<>(1);
        naryRoot.children.add(new NaryTree.Node<>(3));
        naryRoot.children.add(new NaryTree.Node<>(2));
        naryRoot.children.add(new NaryTree.Node<>(4));
        naryRoot.children.get(0).children.add(new NaryTree.Node<>(5));
        naryRoot.children.get(0).children.add(new NaryTree.Node<>(6));
        
        System.out.println("N-ary preorder: " + NaryTree.preorder(naryRoot));
        System.out.println("N-ary level order: " + NaryTree.levelOrder(naryRoot));
        System.out.println("N-ary max depth: " + NaryTree.maxDepth(naryRoot));
    }
}
