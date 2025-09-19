import java.util.Scanner;

public class PerimeterOfEquilateralTriangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the side length of the equilateral triangle: ");
        double side = sc.nextDouble();
        
        // Perimeter of equilateral triangle = 3 * side
        double perimeter = 3 * side;
        
        System.out.println("Perimeter of equilateral triangle: " + perimeter);
        
        sc.close();
    }
}