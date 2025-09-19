import java.util.Scanner;

public class AreaOfRhombus {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the first diagonal of the rhombus: ");
        double diagonal1 = sc.nextDouble();
        
        System.out.print("Enter the second diagonal of the rhombus: ");
        double diagonal2 = sc.nextDouble();
        
        // Area of rhombus = (1/2) * d1 * d2
        double area = 0.5 * diagonal1 * diagonal2;
        
        System.out.println("Area of rhombus: " + area);
        
        sc.close();
    }
}