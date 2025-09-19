import java.util.Scanner;

public class AreaOfTriangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base of the triangle: ");
        double base = sc.nextDouble();
        
        System.out.print("Enter the height of the triangle: ");
        double height = sc.nextDouble();
        
        // Area of triangle = (1/2) * base * height
        double area = 0.5 * base * height;
        
        System.out.println("Area of triangle with base " + base + " and height " + height + " is: " + area);
        
        sc.close();
    }
}