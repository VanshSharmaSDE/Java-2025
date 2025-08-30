package Algorithms.DataStructures.Trees;

/**
 * Segment Tree Implementation
 * Efficient for range queries and updates
 */
public class SegmentTree {
    
    private int[] tree;
    private int[] lazy;
    private int n;
    
    public SegmentTree(int[] arr) {
        n = arr.length;
        tree = new int[4 * n];
        lazy = new int[4 * n];
        build(arr, 0, 0, n - 1);
    }
    
    /**
     * Build segment tree
     * Time Complexity: O(n)
     */
    private void build(int[] arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
    
    /**
     * Range sum query
     * Time Complexity: O(log n)
     */
    public int rangeSum(int left, int right) {
        return rangeSum(0, 0, n - 1, left, right);
    }
    
    private int rangeSum(int node, int start, int end, int left, int right) {
        if (right < start || end < left) {
            return 0;
        }
        
        if (left <= start && end <= right) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        return rangeSum(2 * node + 1, start, mid, left, right) +
               rangeSum(2 * node + 2, mid + 1, end, left, right);
    }
    
    /**
     * Point update
     * Time Complexity: O(log n)
     */
    public void update(int idx, int val) {
        update(0, 0, n - 1, idx, val);
    }
    
    private void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2 * node + 1, start, mid, idx, val);
            } else {
                update(2 * node + 2, mid + 1, end, idx, val);
            }
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
    
    /**
     * Range update with lazy propagation
     * Time Complexity: O(log n)
     */
    public void rangeUpdate(int left, int right, int val) {
        rangeUpdate(0, 0, n - 1, left, right, val);
    }
    
    private void rangeUpdate(int node, int start, int end, int left, int right, int val) {
        if (lazy[node] != 0) {
            tree[node] += lazy[node] * (end - start + 1);
            if (start != end) {
                lazy[2 * node + 1] += lazy[node];
                lazy[2 * node + 2] += lazy[node];
            }
            lazy[node] = 0;
        }
        
        if (start > end || start > right || end < left) {
            return;
        }
        
        if (start >= left && end <= right) {
            tree[node] += val * (end - start + 1);
            if (start != end) {
                lazy[2 * node + 1] += val;
                lazy[2 * node + 2] += val;
            }
            return;
        }
        
        int mid = (start + end) / 2;
        rangeUpdate(2 * node + 1, start, mid, left, right, val);
        rangeUpdate(2 * node + 2, mid + 1, end, left, right, val);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    /**
     * Range sum query with lazy propagation
     */
    public int rangeSumLazy(int left, int right) {
        return rangeSumLazy(0, 0, n - 1, left, right);
    }
    
    private int rangeSumLazy(int node, int start, int end, int left, int right) {
        if (start > end || start > right || end < left) {
            return 0;
        }
        
        if (lazy[node] != 0) {
            tree[node] += lazy[node] * (end - start + 1);
            if (start != end) {
                lazy[2 * node + 1] += lazy[node];
                lazy[2 * node + 2] += lazy[node];
            }
            lazy[node] = 0;
        }
        
        if (start >= left && end <= right) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        return rangeSumLazy(2 * node + 1, start, mid, left, right) +
               rangeSumLazy(2 * node + 2, mid + 1, end, left, right);
    }
    
    /**
     * Range minimum query segment tree
     */
    public static class RangeMinimumQuery {
        private int[] tree;
        private int n;
        
        public RangeMinimumQuery(int[] arr) {
            n = arr.length;
            tree = new int[4 * n];
            build(arr, 0, 0, n - 1);
        }
        
        private void build(int[] arr, int node, int start, int end) {
            if (start == end) {
                tree[node] = arr[start];
            } else {
                int mid = (start + end) / 2;
                build(arr, 2 * node + 1, start, mid);
                build(arr, 2 * node + 2, mid + 1, end);
                tree[node] = Math.min(tree[2 * node + 1], tree[2 * node + 2]);
            }
        }
        
        public int rangeMin(int left, int right) {
            return rangeMin(0, 0, n - 1, left, right);
        }
        
        private int rangeMin(int node, int start, int end, int left, int right) {
            if (right < start || end < left) {
                return Integer.MAX_VALUE;
            }
            
            if (left <= start && end <= right) {
                return tree[node];
            }
            
            int mid = (start + end) / 2;
            return Math.min(rangeMin(2 * node + 1, start, mid, left, right),
                           rangeMin(2 * node + 2, mid + 1, end, left, right));
        }
        
        public void update(int idx, int val) {
            update(0, 0, n - 1, idx, val);
        }
        
        private void update(int node, int start, int end, int idx, int val) {
            if (start == end) {
                tree[node] = val;
            } else {
                int mid = (start + end) / 2;
                if (idx <= mid) {
                    update(2 * node + 1, start, mid, idx, val);
                } else {
                    update(2 * node + 2, mid + 1, end, idx, val);
                }
                tree[node] = Math.min(tree[2 * node + 1], tree[2 * node + 2]);
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Segment Tree Demo:");
        System.out.println("==================");
        
        int[] arr = {1, 3, 5, 7, 9, 11};
        SegmentTree st = new SegmentTree(arr);
        
        // Range sum queries
        System.out.println("Original array: " + java.util.Arrays.toString(arr));
        System.out.println("Range sum [1, 3]: " + st.rangeSum(1, 3)); // 3+5+7 = 15
        System.out.println("Range sum [2, 5]: " + st.rangeSum(2, 5)); // 5+7+9+11 = 32
        
        // Point update
        System.out.println("\nUpdating index 1 to 10");
        st.update(1, 10);
        System.out.println("Range sum [1, 3]: " + st.rangeSum(1, 3)); // 10+5+7 = 22
        
        // Range update with lazy propagation
        System.out.println("\nRange update [0, 2] by +5");
        st.rangeUpdate(0, 2, 5);
        System.out.println("Range sum [0, 2]: " + st.rangeSumLazy(0, 2)); // (1+5)+(10+5)+(5+5) = 31
        
        // Range minimum query demo
        System.out.println("\nRange Minimum Query Demo:");
        int[] arr2 = {1, 3, 2, 7, 9, 11};
        RangeMinimumQuery rmq = new RangeMinimumQuery(arr2);
        
        System.out.println("Array: " + java.util.Arrays.toString(arr2));
        System.out.println("Range min [1, 4]: " + rmq.rangeMin(1, 4)); // min(3,2,7,9) = 2
        System.out.println("Range min [0, 5]: " + rmq.rangeMin(0, 5)); // min(1,3,2,7,9,11) = 1
        
        rmq.update(2, 0);
        System.out.println("After updating index 2 to 0:");
        System.out.println("Range min [1, 4]: " + rmq.rangeMin(1, 4)); // min(3,0,7,9) = 0
    }
}
