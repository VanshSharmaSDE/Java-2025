import java.util.Scanner;

public class PerimeterOfParallelogram {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the first side length: ");
        double side1 = sc.nextDouble();
        
        System.out.print("Enter the second side length: ");
        double side2 = sc.nextDouble();
        
        // Perimeter of parallelogram = 2 * (side1 + side2)
        double perimeter = 2 * (side1 + side2);
        
        System.out.println("Perimeter of parallelogram: " + perimeter);
        
        sc.close();
    }
}