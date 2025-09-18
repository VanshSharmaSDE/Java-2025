import java.util.*;

public class Question5{
    public static void main(String[] args) {
        sum();
    }

    static void sum(){
        Scanner sc = new Scanner(System.in);
        String input;
        int sum = 0;
        while(true){
            System.out.print("Enter value (or 'x' to stop): ");
            input = sc.next();
            if(input.equals("x")){
                break;
            }
            try {
                int num = Integer.parseInt(input);
                sum = sum + num;
            } catch (NumberFormatException e) {
                System.out.println("Invalid input. Please enter a number or 'x' to stop.");
            }
        }
        System.out.println("Sum: " + sum);
    }
}