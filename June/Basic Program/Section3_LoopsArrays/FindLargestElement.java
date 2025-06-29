import java.util.Scanner;

public class FindLargestElement {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        int largest = findLargest(array);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Largest element: " + largest);
        
        scanner.close();
    }
    
    public static int findLargest(int[] arr) {
        if (arr.length == 0) {
            throw new IllegalArgumentException("Array is empty");
        }
        
        int largest = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > largest) {
                largest = arr[i];
            }
        }
        return largest;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
