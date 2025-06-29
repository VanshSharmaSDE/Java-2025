import java.util.Scanner;

public class LeftRotateArray {
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
        
        leftRotateByOne(array);
        
        System.out.println("Array after left rotation by one position: ");
        printArray(array);
        
        scanner.close();
    }
    
    public static void leftRotateByOne(int[] arr) {
        if (arr.length <= 1) {
            return;
        }
        
        int first = arr[0];
        
        // Shift all elements to the left
        for (int i = 0; i < arr.length - 1; i++) {
            arr[i] = arr[i + 1];
        }
        
        // Place the first element at the end
        arr[arr.length - 1] = first;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
