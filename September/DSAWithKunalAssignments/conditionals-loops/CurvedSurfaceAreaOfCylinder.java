import java.util.Scanner;

public class CurvedSurfaceAreaOfCylinder {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the cylinder: ");
        double radius = sc.nextDouble();
        
        System.out.print("Enter the height of the cylinder: ");
        double height = sc.nextDouble();
        
        // Curved Surface Area of cylinder = 2 * π * r * h
        double curvedSurfaceArea = 2 * Math.PI * radius * height;
        
        System.out.println("Curved Surface Area of cylinder: " + String.format("%.2f", curvedSurfaceArea));
        
        sc.close();
    }
}