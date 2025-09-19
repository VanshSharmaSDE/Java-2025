import java.util.Scanner;

public class VolumeOfPrism {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base area of the prism: ");
        double baseArea = sc.nextDouble();
        
        System.out.print("Enter the height of the prism: ");
        double height = sc.nextDouble();
        
        // Volume of prism = base area * height
        double volume = baseArea * height;
        
        System.out.println("Volume of prism: " + volume);
        
        sc.close();
    }
}