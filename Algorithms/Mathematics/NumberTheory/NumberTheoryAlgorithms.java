package Algorithms.Mathematics.NumberTheory;

/**
 * Number Theory Algorithms
 */
public class NumberTheoryAlgorithms {
    
    /**
     * Calculate Greatest Common Divisor using Euclidean Algorithm
     * @param a First number
     * @param b Second number
     * @return GCD of a and b
     */
    public static int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }
    
    /**
     * Calculate Least Common Multiple
     * @param a First number
     * @param b Second number
     * @return LCM of a and b
     */
    public static int lcm(int a, int b) {
        return Math.abs(a * b) / gcd(a, b);
    }
    
    /**
     * Check if a number is prime
     * @param n Number to check
     * @return true if prime, false otherwise
     */
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Sieve of Eratosthenes to find all primes up to n
     * @param n Upper limit
     * @return Array of booleans where true indicates prime
     */
    public static boolean[] sieveOfEratosthenes(int n) {
        boolean[] prime = new boolean[n + 1];
        for (int i = 0; i <= n; i++) {
            prime[i] = true;
        }
        prime[0] = prime[1] = false;
        
        for (int p = 2; p * p <= n; p++) {
            if (prime[p]) {
                for (int i = p * p; i <= n; i += p) {
                    prime[i] = false;
                }
            }
        }
        return prime;
    }
    
    /**
     * Calculate factorial of n
     * @param n Number
     * @return n!
     */
    public static long factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    
    /**
     * Calculate Fibonacci number at position n
     * @param n Position
     * @return Fibonacci number
     */
    public static long fibonacci(int n) {
        if (n <= 1) return n;
        
        long a = 0, b = 1, c;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
    
    /**
     * Fast exponentiation using binary exponentiation
     * @param base Base number
     * @param exp Exponent
     * @param mod Modulus (use 1 for no modulus)
     * @return base^exp mod mod
     */
    public static long fastPower(long base, long exp, long mod) {
        long result = 1;
        base = base % mod;
        
        while (exp > 0) {
            if (exp % 2 == 1) {
                result = (result * base) % mod;
            }
            exp = exp >> 1;
            base = (base * base) % mod;
        }
        return result;
    }
    
    /**
     * Extended Euclidean Algorithm
     * Finds x, y such that ax + by = gcd(a, b)
     * @param a First number
     * @param b Second number
     * @return Array [gcd, x, y]
     */
    public static int[] extendedGCD(int a, int b) {
        if (a == 0) {
            return new int[]{b, 0, 1};
        }
        
        int[] result = extendedGCD(b % a, a);
        int gcd = result[0];
        int x1 = result[1];
        int y1 = result[2];
        
        int x = y1 - (b / a) * x1;
        int y = x1;
        
        return new int[]{gcd, x, y};
    }
    
    public static void main(String[] args) {
        System.out.println("Number Theory Algorithms:");
        System.out.println("========================");
        
        // GCD and LCM
        int a = 48, b = 18;
        System.out.println("GCD(" + a + ", " + b + ") = " + gcd(a, b));
        System.out.println("LCM(" + a + ", " + b + ") = " + lcm(a, b));
        
        // Prime checking
        int num = 17;
        System.out.println(num + " is prime: " + isPrime(num));
        
        // Factorial
        int factNum = 5;
        System.out.println(factNum + "! = " + factorial(factNum));
        
        // Fibonacci
        int fibNum = 10;
        System.out.println("Fibonacci(" + fibNum + ") = " + fibonacci(fibNum));
        
        // Fast exponentiation
        long base = 2, exp = 10, mod = 1000;
        System.out.println(base + "^" + exp + " mod " + mod + " = " + fastPower(base, exp, mod));
        
        // Extended GCD
        int[] extGCD = extendedGCD(35, 15);
        System.out.println("Extended GCD(35, 15): gcd=" + extGCD[0] + ", x=" + extGCD[1] + ", y=" + extGCD[2]);
        
        // Sieve of Eratosthenes
        System.out.println("\nPrimes up to 30:");
        boolean[] primes = sieveOfEratosthenes(30);
        for (int i = 2; i <= 30; i++) {
            if (primes[i]) {
                System.out.print(i + " ");
            }
        }
        System.out.println();
    }
}
