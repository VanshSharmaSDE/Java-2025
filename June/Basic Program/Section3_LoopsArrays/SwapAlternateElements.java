import java.util.Scanner;

public class SwapAlternateElements {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        System.out.println("Original array: ");
        printArray(array);
        
        swapAlternateElements(array);
        
        System.out.println("Array after swapping alternate elements: ");
        printArray(array);
        
        scanner.close();
    }
    
    public static void swapAlternateElements(int[] arr) {
        for (int i = 0; i < arr.length - 1; i += 2) {
            // Swap elements at positions i and i+1
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
