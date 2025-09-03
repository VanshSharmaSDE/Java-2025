public class CountTheDigit{
    public static void main(String[] args) {
        int num = 437000000;
        int count = 0;
        // while(num > 0){
        //     num = num / 10;
        //     count ++;
        // }
        count = countDigit(num);
        System.out.println(count);
    }

    static int countDigit(int digit){
        int count = 0;
        while(digit > 0){
          digit = digit / 10;
          count++;
        }
        return count;
    }
}