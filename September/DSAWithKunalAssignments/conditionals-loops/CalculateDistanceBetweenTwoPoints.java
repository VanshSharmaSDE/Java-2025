import java.util.Scanner;

public class CalculateDistanceBetweenTwoPoints {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.println("Enter coordinates of first point:");
        System.out.print("x1: ");
        double x1 = sc.nextDouble();
        System.out.print("y1: ");
        double y1 = sc.nextDouble();
        
        System.out.println("Enter coordinates of second point:");
        System.out.print("x2: ");
        double x2 = sc.nextDouble();
        System.out.print("y2: ");
        double y2 = sc.nextDouble();
        
        // Distance formula: √[(x2-x1)² + (y2-y1)²]
        double distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        
        System.out.println("Distance between points (" + x1 + ", " + y1 + ") and (" + x2 + ", " + y2 + ") is: " + String.format("%.2f", distance));
        
        sc.close();
    }
}