import java.util.Scanner;

public class PerimeterOfRhombus {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the side length of the rhombus: ");
        double side = sc.nextDouble();
        
        // Perimeter of rhombus = 4 * side
        double perimeter = 4 * side;
        
        System.out.println("Perimeter of rhombus: " + perimeter);
        
        sc.close();
    }
}