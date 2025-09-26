import java.util.*;

public class ReversePrefix{
    public static void main(String[] args) {
        String str = "abcdef";
        char ch = 'd';
        System.out.println(reversePrefix(str,ch));
    }

     static String reversePrefix(String word, char ch) {
        StringBuilder str = new StringBuilder();
        int index = word.indexOf(ch);
        char[] chr = word.toCharArray();
        int left = 0;
        int right = index;
        while(left < right) {
            char temp = chr[left];
            chr[left] = chr[right];
            chr[right] = temp;
            left++;
            right--;
        }
        return new String(chr);
    }
}