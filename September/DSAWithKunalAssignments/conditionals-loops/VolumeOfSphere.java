import java.util.Scanner;

public class VolumeOfSphere {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the radius of the sphere: ");
        double radius = sc.nextDouble();
        
        // Volume of sphere = (4/3) * π * r³
        double volume = (4.0/3.0) * Math.PI * radius * radius * radius;
        
        System.out.println("Volume of sphere: " + String.format("%.2f", volume));
        
        sc.close();
    }
}