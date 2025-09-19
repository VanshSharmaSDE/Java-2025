import java.util.Scanner;

public class AreaOfParallelogram {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base of the parallelogram: ");
        double base = sc.nextDouble();
        
        System.out.print("Enter the height of the parallelogram: ");
        double height = sc.nextDouble();
        
        // Area of parallelogram = base * height
        double area = base * height;
        
        System.out.println("Area of parallelogram: " + area);
        
        sc.close();
    }
}