import java.util.Scanner;

public class PerimeterOfRectangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the length of the rectangle: ");
        double length = sc.nextDouble();
        
        System.out.print("Enter the width of the rectangle: ");
        double width = sc.nextDouble();
        
        // Perimeter of rectangle = 2 * (length + width)
        double perimeter = 2 * (length + width);
        
        System.out.println("Perimeter of rectangle: " + perimeter);
        
        sc.close();
    }
}