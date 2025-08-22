public class OABinarySearch{
    public static void main(String[] args) {
        int[] arr = {1,2,3,4,5,6,7,8,9};
        int target = 5;
        int result = OrderAgnosticBinarySearch(arr, target);
    }

    public static int OrderAgnosticBinarySearch(int[] arr, int target) {
       int start = 0;
       int end = arr.length - 1;
       if (arr[start] == target) {
           return start;
       }
       if (arr[end] == target) {
           return end;
       }
       while (start <= end) {
           int mid = (start + end) / 2;
           if (arr[mid] == target) {
               return mid;
           }
           if (arr[start] < arr[end]) { // Ascending order
               if (arr[mid] < target) {
                   start = mid + 1;
               } else {
                   end = mid - 1;
               }
           } else { // Descending order
               if (arr[mid] > target) {
                   start = mid + 1;
               } else {
                   end = mid - 1;
               }
           }
       }
       return -1;
    }

}