import java.util.Scanner;

public class FindAllFactors {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int number = sc.nextInt();
        
        System.out.print("Factors of " + number + " are: ");
        
        for (int i = 1; i <= number; i++) {
            if (number % i == 0) {
                System.out.print(i + " ");
            }
        }
        System.out.println();
        
        sc.close();
    }
}