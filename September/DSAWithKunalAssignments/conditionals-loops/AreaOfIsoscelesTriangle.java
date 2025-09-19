import java.util.Scanner;

public class AreaOfIsoscelesTriangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base of the isosceles triangle: ");
        double base = sc.nextDouble();
        
        System.out.print("Enter the equal side length: ");
        double side = sc.nextDouble();
        
        // Calculate height using Pythagorean theorem: h = √(side² - (base/2)²)
        double height = Math.sqrt((side * side) - ((base * base) / 4));
        
        // Area = (1/2) * base * height
        double area = 0.5 * base * height;
        
        System.out.println("Area of isosceles triangle: " + String.format("%.2f", area));
        
        sc.close();
    }
}