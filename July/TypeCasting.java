import java.util.*;

public class TypeCasting{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the value: ");
        // float number = sc.nextFloat();
        // System.out.println("Value in float: " + number);

        float number = sc.nextFloat();
        number = (int)(number); // Explicit type casting
        System.out.println("Value in int: " + number);

        // Example of narrowing conversion
        // This will truncate the decimal part
        int a = 257;
        byte b = (byte)(a);
        System.out.println("Value in byte: " + b);
    }
}