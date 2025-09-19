import java.util.Scanner;

public class AreaOfCircle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the circle: ");
        double radius = sc.nextDouble();
        
        // Area of circle = π * r²
        double area = Math.PI * radius * radius;
        
        System.out.println("Area of circle with radius " + radius + " is: " + String.format("%.2f", area));
        
        sc.close();
    }
}