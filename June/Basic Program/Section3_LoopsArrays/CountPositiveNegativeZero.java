import java.util.Scanner;

public class CountPositiveNegativeZero {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        int[] counts = countElements(array);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Positive numbers: " + counts[0]);
        System.out.println("Negative numbers: " + counts[1]);
        System.out.println("Zero elements: " + counts[2]);
        
        scanner.close();
    }
    
    public static int[] countElements(int[] arr) {
        int positive = 0, negative = 0, zero = 0;
        
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > 0) {
                positive++;
            } else if (arr[i] < 0) {
                negative++;
            } else {
                zero++;
            }
        }
        
        return new int[]{positive, negative, zero};
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
