public class ArmStrongNumber{
    public static void main(String[] args) {
       int num = 153;
       System.out.println(AM(num));
    }

    static boolean AM(int num){
       int digit = count(num);
       int OG = num;
       int finalResult = 0;
       int temp = 0;
       while(num > 0){
         temp = num % 10;
         finalResult = finalResult + multiply(temp,digit);
         num = num / 10;
       }
       if(finalResult == OG) return true;
       return false;
    }

    static int count(int num){
        int count = 0;
        while(num > 0){
            num = num / 10;
            count++;
        }
        return count;
    }

    static int multiply(int num , int count){
        int result = 1;
        while(count > 0){
            result = result * num;
            count--;
        }
        return result;
    }
}