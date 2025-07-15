import java.util.*;

public class PrimeNumber{
    public static void main (String[] args) {
        int baseNum = 2;
        int flag = 0;
        System.out.println("Enter the number");
        Scanner sc = new Scanner(System.in);
        int inputNum = sc.nextInt();

        //main logic
        if(inputNum==0 || inputNum==1 || inputNum==2){
            System.out.println("Not Prime");
            return;
        }
        else{
            while(baseNum*baseNum <= inputNum){
                if(inputNum % baseNum == 0){
                    flag = 1;
                    break;
                }
                baseNum++;
            }
            if(flag == 1){
                System.out.println("Not Prime");
            }
            else{
                System.out.println("Prime");
            }
        }
    }
}