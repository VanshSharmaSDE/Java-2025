public class RotatedBinarySearchDuplicateElement{
    public static void main(String[] args) {
        int arr = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    }

    static int search(int[] nums, int target) {
        int pivot = findPivot(nums);

        if(pivot == -1){
            return binarySearch(nums , target , 0 , nums.length - 1);
        }

        if(nums[pivot] == target){
            return pivot;
        }
        
        if(target >= nums[0]){
            return binarySearch(nums , target , 0 , pivot - 1);
        } else {
            return binarySearch(nums , target , pivot + 1, nums.length - 1);
        }
    }

     static static int binarySearch(int[] arr, int target, int start , int end) {
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

    static int findPivot(int[] arr){
        int start = 0;
        int end = arr.length - 1;

        while(start <= end){
            int mid = start + (end - start) / 2;

            if(mid < end && arr[mid] > arr[mid + 1]){
                return mid;
            }

            if(mid > start && arr[mid] < arr[mid - 1]){
                return mid;
            }

           if(arr[mid] == arr[start] && arr[mid] == arr[end]){
            
            if(arr[start] > arr[start + 1]){
                return start;
            }
            start++;
            if(arr[end] < arr[end - 1]){
                return end - 1;
            }
            end--;

            else if(arr[start] < arr[mid] || (arr[start] == arr[mid] && arr[mid] > arr[end])){
                start = mid + 1;
            } else{
                end = mid - 1;
            }
           }
        }
            return -1;
    }
}
