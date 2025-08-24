public class BinarySearchWithoutLengthMethod{
    public static void main(String[] args) {
        int[] arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        int target = 12;
        System.out.println(ans(arr, target));
    }

    public static int ans(int[] arr, int target){
       int start = 0;
       int end = 1;

       while(target > end){
         int temp = end + 1;
         end = end + ((end - start) + 1) *2;
         start = temp;
       }

       return binarySearch(arr ,target, start, end);
    }

     public static int binarySearch(int[] arr, int target, int start, int end) {
        while(start <= end){
            // i will see why this is used
            int mid = start + (end - start) / 2;
            if(arr[mid] == target){
                return mid; // Element found
            }else if(arr[mid] < target){
                start = mid + 1;
            }else{
                end = mid - 1;
            }
        }
        return -1; // Element not found
    }
}