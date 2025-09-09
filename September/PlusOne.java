import java.util.*;

public class PlusOne{
    public static void main(String[] args) {
        int[] arr = {9,8,7,6,5,4,3,2,1,0};
        System.out.println(Arrays.toString(plusOne(arr)));
    }

    static int[] plusOne(int[] digits) {
        int temp = 0;
        int[] arr = new int[digits.length + 1];
        for(int i = 0; i < digits.length ; i++){
            temp = temp * 10 + digits[i];
        }
         
         temp = temp + 1;
         int tempLength = length(temp);

         if(tempLength == digits.length){
            for(int i = digits.length - 1; i >= 0 ; i--){
            int temp2 = temp%10;
            digits[i] = temp2;
            temp = temp/10;
        }
        return digits;
     }
        if((tempLength > digits.length)){
           for(int i = arr.length - 1; i >= 0 ; i--){
            int temp2 = temp%10;
            arr[i] = temp2;
            temp = temp/10;
        }
        return arr;
      }
      return digits;
    }

    static int length(int n){
        return String.valueOf(Math.abs(n)).length();
    }
}