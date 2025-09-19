import java.util.Scanner;

public class PerimeterOfCircle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the circle: ");
        double radius = sc.nextDouble();
        
        // Perimeter (Circumference) of circle = 2 * π * r
        double perimeter = 2 * Math.PI * radius;
        
        System.out.println("Perimeter of circle: " + String.format("%.2f", perimeter));
        
        sc.close();
    }
}