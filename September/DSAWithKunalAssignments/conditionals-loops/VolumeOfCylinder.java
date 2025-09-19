import java.util.Scanner;

public class VolumeOfCylinder {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the cylinder: ");
        double radius = sc.nextDouble();
        
        System.out.print("Enter the height of the cylinder: ");
        double height = sc.nextDouble();
        
        // Volume of cylinder = π * r² * h
        double volume = Math.PI * radius * radius * height;
        
        System.out.println("Volume of cylinder: " + String.format("%.2f", volume));
        
        sc.close();
    }
}