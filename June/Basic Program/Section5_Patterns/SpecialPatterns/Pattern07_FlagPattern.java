import java.util.Scanner;

public class Pattern07_FlagPattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the height of flag: ");
        int height = scanner.nextInt();
        System.out.print("Enter the width of flag: ");
        int width = scanner.nextInt();
        
        System.out.println("Flag Pattern:");
        
        // Flag pole
        for (int i = 0; i < height + 5; i++) {
            System.out.print("| ");
            
            // Flag body (only for first 'height' rows)
            if (i < height) {
                for (int j = 0; j < width; j++) {
                    // Create striped pattern
                    if (i % 2 == 0) {
                        System.out.print("* ");
                    } else {
                        System.out.print("- ");
                    }
                }
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
