import java.util.*;

public class SelectionSort{
    public static void main(String[] args) {
        int[] arr = {5,4,3,2,1};
        SS(arr);
        System.out.println(Arrays.toString(arr));
    }

    static void SS(int[] arr){
       for (int i = 0; i < arr.length; i++) {
           int end = arr.length - i - 1;
           int maximum = max(arr, 0 , end);
           swap(arr , maximum , end);
       }
    }

    static int max(int[] arr, int start, int end){
        int Max = start;
        for(int i = start ; i <= end ; i++){
            if(arr[Max] < arr[i]){
                Max = i;
            }
        }
        return Max;
    }

    static void swap(int[] arr , int first , int second){
       int temp = arr[first];
       arr[first] = arr[second];
       arr[second] = temp;
    }
}