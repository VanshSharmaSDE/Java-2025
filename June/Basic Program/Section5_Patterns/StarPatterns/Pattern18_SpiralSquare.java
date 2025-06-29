import java.util.Scanner;

public class Pattern18_SpiralSquare {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size: ");
        int n = scanner.nextInt();
        
        System.out.println("Spiral Square Pattern:");
        char[][] pattern = new char[n][n];
        
        // Initialize with spaces
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                pattern[i][j] = ' ';
            }
        }
        
        // Fill the spiral
        int top = 0, bottom = n - 1, left = 0, right = n - 1;
        
        while (top <= bottom && left <= right) {
            // Fill top row
            for (int i = left; i <= right; i++) {
                pattern[top][i] = '*';
            }
            top++;
            
            // Fill right column
            for (int i = top; i <= bottom; i++) {
                pattern[i][right] = '*';
            }
            right--;
            
            // Fill bottom row
            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    pattern[bottom][i] = '*';
                }
                bottom--;
            }
            
            // Fill left column
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    pattern[i][left] = '*';
                }
                left++;
            }
        }
        
        // Print the pattern
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(pattern[i][j] + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
