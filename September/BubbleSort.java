import java.util.*;

public class BubbleSort{
    public static void main(String[] args) {
        int[] arr = {1,2,3,4,5,6,7,8,9,10};
        BS(arr);
        System.out.println(Arrays.toString(arr));
    }

    static void BS(int[] arr){
        boolean swapped;
        for (int i = 0; i < arr.length; i++) {
            swapped = false;
            for (int j = 1; j < arr.length - i; j++) {
                if(arr[j - 1] > arr[j]){
                    int temp = arr[j - 1];
                    arr[j - 1] = arr[j];
                    arr[j] = temp;
                }
                swapped = true;
            }
            if(!swapped) break;
        }
    }
}