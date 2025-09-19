import java.util.Scanner;

public class AreaOfRectangle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the length of the rectangle: ");
        double length = sc.nextDouble();
        
        System.out.print("Enter the width of the rectangle: ");
        double width = sc.nextDouble();
        
        // Area of rectangle = length * width
        double area = length * width;
        
        System.out.println("Area of rectangle with length " + length + " and width " + width + " is: " + area);
        
        sc.close();
    }
}