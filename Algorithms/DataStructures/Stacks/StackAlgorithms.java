package Algorithms.DataStructures.Stacks;

import java.util.Stack;

/**
 * Stack-based Algorithms and Applications
 */
public class StackAlgorithms {
    
    /**
     * Check if parentheses are balanced
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static boolean isBalancedParentheses(String expression) {
        Stack<Character> stack = new Stack<>();
        
        for (char ch : expression.toCharArray()) {
            if (ch == '(' || ch == '[' || ch == '{') {
                stack.push(ch);
            } else if (ch == ')' || ch == ']' || ch == '}') {
                if (stack.isEmpty()) {
                    return false;
                }
                
                char top = stack.pop();
                if (!isMatchingPair(top, ch)) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
    
    private static boolean isMatchingPair(char opening, char closing) {
        return (opening == '(' && closing == ')') ||
               (opening == '[' && closing == ']') ||
               (opening == '{' && closing == '}');
    }
    
    /**
     * Convert infix expression to postfix
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static String infixToPostfix(String infix) {
        StringBuilder postfix = new StringBuilder();
        Stack<Character> stack = new Stack<>();
        
        for (char ch : infix.toCharArray()) {
            if (Character.isLetterOrDigit(ch)) {
                postfix.append(ch);
            } else if (ch == '(') {
                stack.push(ch);
            } else if (ch == ')') {
                while (!stack.isEmpty() && stack.peek() != '(') {
                    postfix.append(stack.pop());
                }
                stack.pop(); // Remove '('
            } else if (isOperator(ch)) {
                while (!stack.isEmpty() && precedence(ch) <= precedence(stack.peek())) {
                    postfix.append(stack.pop());
                }
                stack.push(ch);
            }
        }
        
        while (!stack.isEmpty()) {
            postfix.append(stack.pop());
        }
        
        return postfix.toString();
    }
    
    /**
     * Evaluate postfix expression
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static int evaluatePostfix(String postfix) {
        Stack<Integer> stack = new Stack<>();
        
        for (char ch : postfix.toCharArray()) {
            if (Character.isDigit(ch)) {
                stack.push(ch - '0');
            } else if (isOperator(ch)) {
                int operand2 = stack.pop();
                int operand1 = stack.pop();
                int result = performOperation(operand1, operand2, ch);
                stack.push(result);
            }
        }
        
        return stack.pop();
    }
    
    /**
     * Find next greater element for each element in array
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static int[] nextGreaterElement(int[] arr) {
        int n = arr.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        // Initialize result array with -1
        for (int i = 0; i < n; i++) {
            result[i] = -1;
        }
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
                result[stack.pop()] = arr[i];
            }
            stack.push(i);
        }
        
        return result;
    }
    
    /**
     * Find largest rectangular area in histogram
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int n = heights.length;
        
        for (int i = 0; i <= n; i++) {
            int currentHeight = (i == n) ? 0 : heights[i];
            
            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            
            stack.push(i);
        }
        
        return maxArea;
    }
    
    /**
     * Implement min stack that supports getMin() in O(1)
     */
    public static class MinStack {
        private Stack<Integer> stack;
        private Stack<Integer> minStack;
        
        public MinStack() {
            stack = new Stack<>();
            minStack = new Stack<>();
        }
        
        public void push(int val) {
            stack.push(val);
            if (minStack.isEmpty() || val <= minStack.peek()) {
                minStack.push(val);
            }
        }
        
        public void pop() {
            if (stack.peek().equals(minStack.peek())) {
                minStack.pop();
            }
            stack.pop();
        }
        
        public int top() {
            return stack.peek();
        }
        
        public int getMin() {
            return minStack.peek();
        }
    }
    
    /**
     * Stock span problem - find span of stock prices
     * Time Complexity: O(n)
     * Space Complexity: O(n)
     */
    public static int[] stockSpan(int[] prices) {
        int n = prices.length;
        int[] span = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && prices[stack.peek()] <= prices[i]) {
                stack.pop();
            }
            
            span[i] = stack.isEmpty() ? i + 1 : i - stack.peek();
            stack.push(i);
        }
        
        return span;
    }
    
    private static boolean isOperator(char ch) {
        return ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^';
    }
    
    private static int precedence(char operator) {
        switch (operator) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
                return 2;
            case '^':
                return 3;
            default:
                return -1;
        }
    }
    
    private static int performOperation(int operand1, int operand2, char operator) {
        switch (operator) {
            case '+': return operand1 + operand2;
            case '-': return operand1 - operand2;
            case '*': return operand1 * operand2;
            case '/': return operand1 / operand2;
            case '^': return (int) Math.pow(operand1, operand2);
            default: throw new IllegalArgumentException("Invalid operator: " + operator);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Stack Algorithms Demo:");
        System.out.println("======================");
        
        // Balanced parentheses
        String expr1 = "((()))";
        String expr2 = "([{}])";
        String expr3 = "((]";
        
        System.out.println("Balanced Parentheses:");
        System.out.println(expr1 + " -> " + isBalancedParentheses(expr1));
        System.out.println(expr2 + " -> " + isBalancedParentheses(expr2));
        System.out.println(expr3 + " -> " + isBalancedParentheses(expr3));
        
        // Infix to postfix
        String infix = "A+B*C-D";
        String postfix = infixToPostfix(infix);
        System.out.println("\nInfix to Postfix:");
        System.out.println("Infix: " + infix);
        System.out.println("Postfix: " + postfix);
        
        // Evaluate postfix
        String postfixExpr = "23+4*";
        int result = evaluatePostfix(postfixExpr);
        System.out.println("\nPostfix Evaluation:");
        System.out.println("Expression: " + postfixExpr);
        System.out.println("Result: " + result);
        
        // Next greater element
        int[] arr = {4, 5, 2, 25, 7, 8};
        int[] nge = nextGreaterElement(arr);
        System.out.println("\nNext Greater Element:");
        System.out.print("Array: ");
        for (int num : arr) System.out.print(num + " ");
        System.out.print("\nNGE:   ");
        for (int num : nge) System.out.print(num + " ");
        System.out.println();
        
        // Largest rectangle in histogram
        int[] heights = {2, 1, 5, 6, 2, 3};
        int maxArea = largestRectangleArea(heights);
        System.out.println("\nLargest Rectangle in Histogram:");
        System.out.print("Heights: ");
        for (int h : heights) System.out.print(h + " ");
        System.out.println("\nMax Area: " + maxArea);
        
        // Min stack
        System.out.println("\nMin Stack Demo:");
        MinStack minStack = new MinStack();
        minStack.push(3);
        minStack.push(5);
        System.out.println("Min: " + minStack.getMin()); // 3
        minStack.push(2);
        minStack.push(1);
        System.out.println("Min: " + minStack.getMin()); // 1
        minStack.pop();
        System.out.println("Min: " + minStack.getMin()); // 2
        
        // Stock span
        int[] prices = {100, 80, 60, 70, 60, 75, 85};
        int[] span = stockSpan(prices);
        System.out.println("\nStock Span Problem:");
        System.out.print("Prices: ");
        for (int p : prices) System.out.print(p + " ");
        System.out.print("\nSpan:   ");
        for (int s : span) System.out.print(s + " ");
        System.out.println();
    }
}
