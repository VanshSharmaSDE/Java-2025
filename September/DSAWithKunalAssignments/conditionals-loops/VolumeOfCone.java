import java.util.Scanner;

public class VolumeOfCone {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the cone: ");
        double radius = sc.nextDouble();
        
        System.out.print("Enter the height of the cone: ");
        double height = sc.nextDouble();
        
        // Volume of cone = (1/3) * π * r² * h
        double volume = (1.0/3.0) * Math.PI * radius * radius * height;
        
        System.out.println("Volume of cone: " + String.format("%.2f", volume));
        
        sc.close();
    }
}