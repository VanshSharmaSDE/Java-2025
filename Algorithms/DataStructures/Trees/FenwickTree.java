package Algorithms.DataStructures.Trees;

/**
 * Fenwick Tree (Binary Indexed Tree) Implementation
 * Efficient for prefix sum queries and updates
 */
public class FenwickTree {
    
    private int[] tree;
    private int n;
    
    public FenwickTree(int size) {
        n = size;
        tree = new int[n + 1];
    }
    
    public FenwickTree(int[] arr) {
        n = arr.length;
        tree = new int[n + 1];
        
        for (int i = 0; i < n; i++) {
            update(i, arr[i]);
        }
    }
    
    /**
     * Update value at index
     * Time Complexity: O(log n)
     */
    public void update(int idx, int delta) {
        for (int i = idx + 1; i <= n; i += i & (-i)) {
            tree[i] += delta;
        }
    }
    
    /**
     * Get prefix sum [0, idx]
     * Time Complexity: O(log n)
     */
    public int prefixSum(int idx) {
        int sum = 0;
        for (int i = idx + 1; i > 0; i -= i & (-i)) {
            sum += tree[i];
        }
        return sum;
    }
    
    /**
     * Get range sum [left, right]
     * Time Complexity: O(log n)
     */
    public int rangeSum(int left, int right) {
        if (left == 0) {
            return prefixSum(right);
        }
        return prefixSum(right) - prefixSum(left - 1);
    }
    
    /**
     * 2D Fenwick Tree for 2D range sum queries
     */
    public static class FenwickTree2D {
        private int[][] tree;
        private int rows, cols;
        
        public FenwickTree2D(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            tree = new int[rows + 1][cols + 1];
        }
        
        public void update(int row, int col, int delta) {
            for (int i = row + 1; i <= rows; i += i & (-i)) {
                for (int j = col + 1; j <= cols; j += j & (-j)) {
                    tree[i][j] += delta;
                }
            }
        }
        
        public int prefixSum(int row, int col) {
            int sum = 0;
            for (int i = row + 1; i > 0; i -= i & (-i)) {
                for (int j = col + 1; j > 0; j -= j & (-j)) {
                    sum += tree[i][j];
                }
            }
            return sum;
        }
        
        public int rangeSum(int row1, int col1, int row2, int col2) {
            return prefixSum(row2, col2) 
                 - prefixSum(row1 - 1, col2) 
                 - prefixSum(row2, col1 - 1) 
                 + prefixSum(row1 - 1, col1 - 1);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Fenwick Tree Demo:");
        System.out.println("==================");
        
        int[] arr = {1, 3, 5, 7, 9, 11};
        FenwickTree ft = new FenwickTree(arr);
        
        System.out.println("Original array: " + java.util.Arrays.toString(arr));
        
        // Prefix sum queries
        System.out.println("Prefix sum [0, 2]: " + ft.prefixSum(2)); // 1+3+5 = 9
        System.out.println("Prefix sum [0, 4]: " + ft.prefixSum(4)); // 1+3+5+7+9 = 25
        
        // Range sum queries
        System.out.println("Range sum [1, 3]: " + ft.rangeSum(1, 3)); // 3+5+7 = 15
        System.out.println("Range sum [2, 5]: " + ft.rangeSum(2, 5)); // 5+7+9+11 = 32
        
        // Update operation
        System.out.println("\nUpdating index 2 by +10 (5 -> 15)");
        ft.update(2, 10);
        System.out.println("Range sum [1, 3]: " + ft.rangeSum(1, 3)); // 3+15+7 = 25
        System.out.println("Prefix sum [0, 4]: " + ft.prefixSum(4)); // 1+3+15+7+9 = 35
        
        // 2D Fenwick Tree demo
        System.out.println("\n2D Fenwick Tree Demo:");
        FenwickTree2D ft2d = new FenwickTree2D(4, 4);
        
        // Update some cells
        ft2d.update(1, 1, 5);
        ft2d.update(2, 2, 3);
        ft2d.update(3, 3, 7);
        
        System.out.println("Range sum [0,0] to [2,2]: " + ft2d.rangeSum(0, 0, 2, 2)); // 5+3 = 8
        System.out.println("Range sum [1,1] to [3,3]: " + ft2d.rangeSum(1, 1, 3, 3)); // 5+3+7 = 15
    }
}
