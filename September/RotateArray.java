import java.util.*;

public class RotateArray{
    public static void main(String[] args) {
        int[] nums = {1,2,3,4,5,6,7};
        int k = 54944;
        rotate(nums,k); 
        System.out.println(Arrays.toString(nums));
    }

    static void rotate(int[] nums, int l) {
        int[] arr = new int[nums.length];

        for(int i = 0 ; i < l ; i++){
            arr[0] = nums[nums.length - 1];
            for(int j = 0 ; j < nums.length - 1 ; j++){
                arr[j + 1] = nums[j];
                // System.out.println(arr[j + 1]);
            }
            for(int k = 0 ; k < nums.length ; k++){
                nums[k] = arr[k];
        }
        }
    }
}