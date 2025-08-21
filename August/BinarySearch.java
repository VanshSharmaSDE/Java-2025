public class BinarySearch{
    public static void main(String[] args) {
        int arr[] = {1 ,2 ,3 ,4 ,5};
        int target = 0;
        int result = binarySearch(arr, target);
        if(result != -1){
            System.out.println("Element found at index: " + result);
        }else{
            System.out.println("Element not found");
        }
    }

    public static int binarySearch(int[] arr, int target) {
        int start = 0;
        int end = arr.length - 1;
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