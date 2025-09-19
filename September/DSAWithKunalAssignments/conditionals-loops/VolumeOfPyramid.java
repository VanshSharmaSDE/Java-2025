import java.util.Scanner;

public class VolumeOfPyramid {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the base area of the pyramid: ");
        double baseArea = sc.nextDouble();
        
        System.out.print("Enter the height of the pyramid: ");
        double height = sc.nextDouble();
        
        // Volume of pyramid = (1/3) * base area * height
        double volume = (1.0/3.0) * baseArea * height;
        
        System.out.println("Volume of pyramid: " + String.format("%.2f", volume));
        
        sc.close();
    }
}