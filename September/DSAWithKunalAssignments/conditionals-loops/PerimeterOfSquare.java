import java.util.Scanner;

public class PerimeterOfSquare {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the side length of the square: ");
        double side = sc.nextDouble();
        
        // Perimeter of square = 4 * side
        double perimeter = 4 * side;
        
        System.out.println("Perimeter of square: " + perimeter);
        
        sc.close();
    }
}