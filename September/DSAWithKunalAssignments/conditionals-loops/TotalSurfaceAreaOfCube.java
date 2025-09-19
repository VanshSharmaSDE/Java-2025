import java.util.Scanner;

public class TotalSurfaceAreaOfCube {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the side length of the cube: ");
        double side = sc.nextDouble();
        
        // Total Surface Area of cube = 6 * side²
        double totalSurfaceArea = 6 * side * side;
        
        System.out.println("Total Surface Area of cube: " + totalSurfaceArea);
        
        sc.close();
    }
}