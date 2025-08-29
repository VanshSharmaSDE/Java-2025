package Algorithms.ComputerScience.DynamicProgramming;

import java.util.*;

/**
 * Dynamic Programming Algorithms
 */
public class DynamicProgrammingAlgorithms {
    
    /**
     * Fibonacci using Dynamic Programming (Bottom-up)
     * @param n Position in Fibonacci sequence
     * @return Fibonacci number at position n
     */
    public static long fibonacciDP(int n) {
        if (n <= 1) return n;
        
        long[] dp = new long[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[n];
    }
    
    /**
     * Fibonacci using memoization (Top-down)
     * @param n Position in Fibonacci sequence
     * @return Fibonacci number at position n
     */
    public static long fibonacciMemo(int n) {
        Map<Integer, Long> memo = new HashMap<>();
        return fibonacciMemoHelper(n, memo);
    }
    
    private static long fibonacciMemoHelper(int n, Map<Integer, Long> memo) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);
        
        long result = fibonacciMemoHelper(n - 1, memo) + fibonacciMemoHelper(n - 2, memo);
        memo.put(n, result);
        return result;
    }
    
    /**
     * 0/1 Knapsack Problem
     * @param weights Array of item weights
     * @param values Array of item values
     * @param capacity Knapsack capacity
     * @return Maximum value that can be obtained
     */
    public static int knapsack01(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]], // Include item
                        dp[i - 1][w] // Exclude item
                    );
                } else {
                    dp[i][w] = dp[i - 1][w]; // Exclude item
                }
            }
        }
        
        return dp[n][capacity];
    }
    
    /**
     * Longest Common Subsequence
     * @param text1 First string
     * @param text2 Second string
     * @return Length of LCS
     */
    public static int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Edit Distance (Levenshtein Distance)
     * @param word1 First word
     * @param word2 Second word
     * @return Minimum edit distance
     */
    public static int editDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        // Initialize base cases
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1]; // No operation needed
                } else {
                    dp[i][j] = 1 + Math.min(
                        Math.min(dp[i - 1][j], dp[i][j - 1]), // Insert or delete
                        dp[i - 1][j - 1] // Replace
                    );
                }
            }
        }
        
        return dp[m][n];
    }
    
    /**
     * Coin Change Problem - Minimum coins needed
     * @param coins Array of coin denominations
     * @param amount Target amount
     * @return Minimum number of coins needed, -1 if impossible
     */
    public static int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }
    
    /**
     * Longest Increasing Subsequence
     * @param nums Array of numbers
     * @return Length of LIS
     */
    public static int longestIncreasingSubsequence(int[] nums) {
        if (nums.length == 0) return 0;
        
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        
        return Arrays.stream(dp).max().orElse(0);
    }
    
    /**
     * Maximum Subarray Sum (Kadane's Algorithm)
     * @param nums Array of numbers
     * @return Maximum sum of contiguous subarray
     */
    public static int maxSubarraySum(int[] nums) {
        int maxSoFar = nums[0];
        int maxEndingHere = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
    
    /**
     * House Robber Problem
     * @param nums Array representing money in each house
     * @return Maximum money that can be robbed
     */
    public static int houseRobber(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        
        return dp[nums.length - 1];
    }
    
    /**
     * Palindrome Partitioning - Minimum cuts needed
     * @param s Input string
     * @return Minimum number of cuts for palindrome partitioning
     */
    public static int minCutPalindrome(String s) {
        int n = s.length();
        boolean[][] isPalindrome = new boolean[n][n];
        
        // Check all substrings for palindrome
        for (int i = 0; i < n; i++) {
            isPalindrome[i][i] = true;
        }
        
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (len == 2) {
                    isPalindrome[i][j] = (s.charAt(i) == s.charAt(j));
                } else {
                    isPalindrome[i][j] = (s.charAt(i) == s.charAt(j)) && isPalindrome[i + 1][j - 1];
                }
            }
        }
        
        // Calculate minimum cuts
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            if (isPalindrome[0][i]) {
                dp[i] = 0;
            } else {
                dp[i] = Integer.MAX_VALUE;
                for (int j = 0; j < i; j++) {
                    if (isPalindrome[j + 1][i]) {
                        dp[i] = Math.min(dp[i], dp[j] + 1);
                    }
                }
            }
        }
        
        return dp[n - 1];
    }
    
    public static void main(String[] args) {
        System.out.println("Dynamic Programming Algorithms:");
        System.out.println("===============================");
        
        // Fibonacci
        int n = 10;
        System.out.println("Fibonacci(" + n + "):");
        System.out.println("DP: " + fibonacciDP(n));
        System.out.println("Memoization: " + fibonacciMemo(n));
        
        // Knapsack Problem
        int[] weights = {10, 20, 30};
        int[] values = {60, 100, 120};
        int capacity = 50;
        System.out.println("\nKnapsack Problem:");
        System.out.println("Weights: " + Arrays.toString(weights));
        System.out.println("Values: " + Arrays.toString(values));
        System.out.println("Capacity: " + capacity);
        System.out.println("Maximum value: " + knapsack01(weights, values, capacity));
        
        // Longest Common Subsequence
        String text1 = "ABCDGH";
        String text2 = "AEDFHR";
        System.out.println("\nLongest Common Subsequence:");
        System.out.println("Text1: " + text1);
        System.out.println("Text2: " + text2);
        System.out.println("LCS length: " + longestCommonSubsequence(text1, text2));
        
        // Edit Distance
        String word1 = "horse";
        String word2 = "ros";
        System.out.println("\nEdit Distance:");
        System.out.println("Word1: " + word1);
        System.out.println("Word2: " + word2);
        System.out.println("Edit distance: " + editDistance(word1, word2));
        
        // Coin Change
        int[] coins = {1, 3, 4};
        int amount = 6;
        System.out.println("\nCoin Change:");
        System.out.println("Coins: " + Arrays.toString(coins));
        System.out.println("Amount: " + amount);
        System.out.println("Minimum coins: " + coinChange(coins, amount));
        
        // Longest Increasing Subsequence
        int[] nums = {10, 9, 2, 5, 3, 7, 101, 18};
        System.out.println("\nLongest Increasing Subsequence:");
        System.out.println("Array: " + Arrays.toString(nums));
        System.out.println("LIS length: " + longestIncreasingSubsequence(nums));
        
        // Maximum Subarray Sum
        int[] subarray = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        System.out.println("\nMaximum Subarray Sum:");
        System.out.println("Array: " + Arrays.toString(subarray));
        System.out.println("Max sum: " + maxSubarraySum(subarray));
        
        // House Robber
        int[] houses = {2, 7, 9, 3, 1};
        System.out.println("\nHouse Robber:");
        System.out.println("Houses: " + Arrays.toString(houses));
        System.out.println("Max money: " + houseRobber(houses));
        
        // Palindrome Partitioning
        String palindromeStr = "aab";
        System.out.println("\nPalindrome Partitioning:");
        System.out.println("String: " + palindromeStr);
        System.out.println("Min cuts: " + minCutPalindrome(palindromeStr));
    }
}
