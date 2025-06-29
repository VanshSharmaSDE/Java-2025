import java.util.Scanner;

public class Pattern13_PrimeNumberTriangle {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the number of rows: ");
        int n = scanner.nextInt();
        
        System.out.println("Prime Number Triangle Pattern:");
        int primeNum = 2;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(primeNum + " ");
                primeNum = getNextPrime(primeNum);
            }
            System.out.println();
        }
        
        scanner.close();
    }
    
    public static boolean isPrime(int num) {
        if (num <= 1) return false;
        if (num <= 3) return true;
        if (num % 2 == 0 || num % 3 == 0) return false;
        
        for (int i = 5; i * i <= num; i += 6) {
            if (num % i == 0 || num % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }
    
    public static int getNextPrime(int num) {
        num++;
        while (!isPrime(num)) {
            num++;
        }
        return num;
    }
}
