import java.util.Scanner;

public class FindElementFrequency {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        System.out.print("Enter element to find frequency: ");
        int element = scanner.nextInt();
        
        int frequency = findFrequency(array, element);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Frequency of " + element + " in the array: " + frequency);
        
        scanner.close();
    }
    
    public static int findFrequency(int[] arr, int element) {
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == element) {
                count++;
            }
        }
        return count;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
