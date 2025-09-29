public class SumOfDigits{
    public static void main(String[] args) {
        int num = 11;
        System.out.println(digit(num));
    }

    static int digit(int n){
        int ans = 0;
        while(n > 0){
          int temp = n % 10;
          ans = ans + temp;
          n = n / 10;
        }
        return ans;
    }
}