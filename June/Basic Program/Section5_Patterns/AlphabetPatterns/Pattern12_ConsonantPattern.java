import java.util.Scanner;

public class Pattern12_ConsonantPattern {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Consonant Pattern:");
        String consonants = "BCDFGHJKLMNPQRSTVWXYZ";
        int index = 0;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(consonants.charAt(index % consonants.length()) + " ");
                index++;
            }
            System.out.println();
        }
        
        scanner.close();
    }
}
