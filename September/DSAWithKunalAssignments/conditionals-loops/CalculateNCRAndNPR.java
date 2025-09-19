import java.util.Scanner;

public class CalculateNCRAndNPR {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter n: ");
        int n = sc.nextInt();
        
        System.out.print("Enter r: ");
        int r = sc.nextInt();
        
        if (r > n || n < 0 || r < 0) {
            System.out.println("Invalid input! n should be >= r and both should be non-negative.");
        } else {
            long nFactorial = factorial(n);
            long rFactorial = factorial(r);
            long nMinusRFactorial = factorial(n - r);
            
            // nCr = n! / (r! * (n-r)!)
            long ncr = nFactorial / (rFactorial * nMinusRFactorial);
            
            // nPr = n! / (n-r)!
            long npr = nFactorial / nMinusRFactorial;
            
            System.out.println("nCr = " + n + "C" + r + " = " + ncr);
            System.out.println("nPr = " + n + "P" + r + " = " + npr);
        }
        
        sc.close();
    }
    
    static long factorial(int num) {
        if (num <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= num; i++) {
            result *= i;
        }
        return result;
    }
}