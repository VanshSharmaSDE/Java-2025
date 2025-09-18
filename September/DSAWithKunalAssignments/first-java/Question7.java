import java.util.Scanner;

public class Question7 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter the number of terms in Fibonacci series: ");
        int n = sc.nextInt();
        
        if (n <= 0) {
            System.out.println("Please enter a positive number.");
        } else if (n == 1) {
            System.out.println("Fibonacci Series: 0");
        } else {
            System.out.print("Fibonacci Series: ");
            
            int first = 0, second = 1;
            System.out.print(first + " " + second + " ");
            
            for (int i = 3; i <= n; i++) {
                int next = first + second;
                System.out.print(next + " ");
                first = second;
                second = next;
            }
            System.out.println();
        }
        
        sc.close();
    }
}