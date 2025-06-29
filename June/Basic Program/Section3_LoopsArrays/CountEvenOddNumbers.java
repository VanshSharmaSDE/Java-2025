import java.util.Scanner;

public class CountEvenOddNumbers {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the size of array: ");
        int size = scanner.nextInt();
        
        int[] array = new int[size];
        
        System.out.println("Enter " + size + " elements:");
        for (int i = 0; i < size; i++) {
            array[i] = scanner.nextInt();
        }
        
        int[] counts = countEvenOdd(array);
        
        System.out.println("Array elements: ");
        printArray(array);
        System.out.println("Even numbers: " + counts[0]);
        System.out.println("Odd numbers: " + counts[1]);
        
        scanner.close();
    }
    
    public static int[] countEvenOdd(int[] arr) {
        int evenCount = 0, oddCount = 0;
        
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] % 2 == 0) {
                evenCount++;
            } else {
                oddCount++;
            }
        }
        
        return new int[]{evenCount, oddCount};
    }
    
    public static void printArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
}
