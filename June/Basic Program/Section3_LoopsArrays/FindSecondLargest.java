import java.util.Scanner;

public class FindSecondLargest {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        int secondLargest = findSecondLargest(array);
        
        System.out.println("Array elements: ");
        printArray(array);
        
        if (secondLargest != Integer.MIN_VALUE) {
            System.out.println("Second largest element: " + secondLargest);
        } else {
            System.out.println("No second largest element found (all elements might be same).");
        }
        
        scanner.close();
    }
    
    public static int findSecondLargest(int[] arr) {
        if (arr.length < 2) {
            return Integer.MIN_VALUE;
        }
        
        int largest = Integer.MIN_VALUE;
        int secondLargest = Integer.MIN_VALUE;
        
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > largest) {
                secondLargest = largest;
                largest = arr[i];
            } else if (arr[i] > secondLargest && arr[i] != largest) {
                secondLargest = arr[i];
            }
        }
        
        return secondLargest;
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
