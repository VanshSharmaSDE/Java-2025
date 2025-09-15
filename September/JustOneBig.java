public class JustOneBig{
    public static void main(String[] args) {
        int arr[] = {0,1,2,3,4};
        System.out.println(JOB(arr));
    }

    static boolean JOB(int[] arr){
        for (int i = 0; i < arr.length - 1; i++) {
            if(arr[i]+1 != arr[i + 1]){
                return false;
            }
        }
        return true;
    }
}