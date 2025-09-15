public class Palindrome{
    public static void main(String[] args) {
        int number = 121;
        System.out.println(isPalindrome(-121));
    }

    static boolean isPalindrome(int x) {
        int originalnumber = x;
        int reverse = 0;
        while(x > 0){
           int temp = x%10;
           reverse = reverse * 10 + temp;
           x = x/10;
        }
        if(originalnumber == reverse){
            return true;
        } 
        System.out.println(reverse);
        return false;
    }
}