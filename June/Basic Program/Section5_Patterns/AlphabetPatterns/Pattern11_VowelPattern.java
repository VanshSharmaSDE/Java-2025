import java.util.Scanner;

public class Pattern11_VowelPattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Vowel Pattern:");
        char[] vowels = {'A', 'E', 'I', 'O', 'U'};
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(vowels[(j-1) % 5] + " ");
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
