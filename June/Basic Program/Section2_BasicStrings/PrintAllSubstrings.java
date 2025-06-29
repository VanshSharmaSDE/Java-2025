import java.util.Scanner;

public class PrintAllSubstrings {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter a string: ");
        String str = scanner.nextLine();
        
        System.out.println("All substrings of \"" + str + "\":");
        printAllSubstrings(str);
        
        scanner.close();
    }
    
    public static void printAllSubstrings(String str) {
        int n = str.length();
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String substring = str.substring(i, j);
                System.out.println(++count + ": " + substring);
            }
        }
        
        System.out.println("Total substrings: " + count);
    }
}
