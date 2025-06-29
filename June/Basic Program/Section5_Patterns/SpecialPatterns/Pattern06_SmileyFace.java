import java.util.Scanner;

public class Pattern06_SmileyFace {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size (recommended 7-15): ");
        int n = scanner.nextInt();
        
        System.out.println("Smiley Face Pattern:");
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Draw circle boundary
                int centerX = n / 2;
                int centerY = n / 2;
                int radius = n / 2 - 1;
                
                int dx = i - centerX;
                int dy = j - centerY;
                int distance = (int) Math.sqrt(dx * dx + dy * dy);
                
                // Eyes
                if ((i == n/3 && (j == n/3 || j == 2*n/3)) ||
                    // Mouth (smile curve)
                    (i == 2*n/3 && j >= n/3 && j <= 2*n/3) ||
                    // Face boundary
                    (distance == radius)) {
                    System.out.print("* ");
                } else {
                    System.out.print("  ");
                }
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
