import java.util.Scanner;

public class AreaOfEquilateralTriangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the side length of the equilateral triangle: ");
        double side = sc.nextDouble();
        
        // Area of equilateral triangle = (√3/4) * side²
        double area = (Math.sqrt(3) / 4) * side * side;
        
        System.out.println("Area of equilateral triangle: " + String.format("%.2f", area));
        
        sc.close();
    }
}