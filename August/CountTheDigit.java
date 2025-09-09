import java.math.BigInteger;
import java.util.Scanner;

public class CountTheDigit{
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("=== Count Digits Program (Any Length) ===");
        System.out.println("Choose an approach:");
        System.out.println("1. Using String approach");
        System.out.println("2. Using long data type");
        System.out.println("3. Using BigInteger for extremely large numbers");
        System.out.println("4. Using predefined Java methods");
        System.out.println("5. Test all approaches with sample numbers");
        System.out.print("Enter your choice (1-5): ");
        
        int choice = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        
        switch(choice) {
            case 1:
                testStringApproach(scanner);
                break;
            case 2:
                testLongApproach(scanner);
                break;
            case 3:
                testBigIntegerApproach(scanner);
                break;
            case 4:
                testPredefinedApproach(scanner);
                break;
            case 5:
                testAllApproaches();
                break;
            default:
                System.out.println("Invalid choice!");
        }
        
        scanner.close();
    }
    
    // Approach 1: String-based counting (handles any length)
    static int countDigitsString(String number) {
        // Remove any non-digit characters except minus sign
        String cleanNumber = number.replaceAll("[^0-9-]", "");
        
        // Handle negative numbers
        if (cleanNumber.startsWith("-")) {
            cleanNumber = cleanNumber.substring(1);
        }
        
        // Handle empty string or zero
        if (cleanNumber.isEmpty() || cleanNumber.equals("0")) {
            return cleanNumber.isEmpty() ? 0 : 1;
        }
        
        // Remove leading zeros
        cleanNumber = cleanNumber.replaceFirst("^0+", "");
        
        return cleanNumber.isEmpty() ? 1 : cleanNumber.length();
    }
    
    // Approach 2: Using long data type (handles up to 19 digits)
    static int countDigitsLong(long number) {
        if (number == 0) return 1;
        
        number = Math.abs(number); // Handle negative numbers
        int count = 0;
        
        while (number > 0) {
            number = number / 10;
            count++;
        }
        
        return count;
    }
    
    // Alternative long approach using logarithm (faster for large numbers)
    static int countDigitsLongLog(long number) {
        if (number == 0) return 1;
        
        number = Math.abs(number);
        return (int) Math.floor(Math.log10(number)) + 1;
    }
    
    // Approach 3: Using BigInteger for extremely large numbers
    static int countDigitsBigInteger(BigInteger number) {
        if (number.equals(BigInteger.ZERO)) return 1;
        
        number = number.abs(); // Handle negative numbers
        return number.toString().length();
    }
    
    // Approach 4: Using predefined Java methods (most concise)
    static int countDigitsPredefined(String number) {
        // Method 1: Using String.valueOf() and length()
        return String.valueOf(Math.abs(Long.parseLong(number.trim()))).length();
    }
    
    static int countDigitsPredefinedInt(int number) {
        // Method 2: Convert to string directly
        return String.valueOf(Math.abs(number)).length();
    }
    
    static int countDigitsPredefinedLong(long number) {
        // Method 3: Using Long.toString()
        return Long.toString(Math.abs(number)).length();
    }
    
    static int countDigitsPredefinedAdvanced(String number) {
        // Method 4: Using regex to count only digits
        return (int) number.chars()
                          .filter(Character::isDigit)
                          .count();
    }
    
    static int countDigitsPredefinedStream(String number) {
        // Method 5: Using Stream API for digit counting
        String cleanNumber = number.replaceAll("[^0-9]", "");
        if (cleanNumber.isEmpty()) return 0;
        
        // Remove leading zeros and count
        return cleanNumber.replaceFirst("^0+", "").length() > 0 ? 
               cleanNumber.replaceFirst("^0+", "").length() : 1;
    }
    
    static int countDigitsPredefinedDecimalFormat(double number) {
        // Method 6: Using DecimalFormat for scientific notation handling
        java.text.DecimalFormat df = new java.text.DecimalFormat("0");
        String formatted = df.format(Math.abs(number));
        return formatted.length();
    }
    
    // Original method (for comparison) - limited to int range
    static int countDigitOriginal(int digit) {
        if (digit == 0) return 1;
        
        digit = Math.abs(digit); // Handle negative numbers
        int count = 0;
        
        while (digit > 0) {
            digit = digit / 10;
            count++;
        }
        
        return count;
    }
    
    // Test methods for each approach
    static void testStringApproach(Scanner scanner) {
        System.out.print("Enter a number (any length): ");
        String input = scanner.nextLine();
        
        int count = countDigitsString(input);
        System.out.println("Number of digits in '" + input + "': " + count);
        
        // Show some examples
        System.out.println("\nTesting with various inputs:");
        String[] testCases = {"0", "123", "-456", "1234567890123456789", "000123", "12.34", "abc123def"};
        
        for (String test : testCases) {
            System.out.println("'" + test + "' -> " + countDigitsString(test) + " digits");
        }
    }
    
    static void testLongApproach(Scanner scanner) {
        System.out.print("Enter a number (up to 19 digits): ");
        
        try {
            long input = scanner.nextLong();
            
            int count1 = countDigitsLong(input);
            int count2 = countDigitsLongLog(input);
            
            System.out.println("Number: " + input);
            System.out.println("Digits (division method): " + count1);
            System.out.println("Digits (logarithm method): " + count2);
            
        } catch (Exception e) {
            System.out.println("Invalid input or number too large for long data type!");
        }
    }
    
    static void testBigIntegerApproach(Scanner scanner) {
        System.out.print("Enter a very large number: ");
        String input = scanner.nextLine();
        
        try {
            BigInteger bigNum = new BigInteger(input);
            int count = countDigitsBigInteger(bigNum);
            
            System.out.println("Number: " + bigNum);
            System.out.println("Number of digits: " + count);
            
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format!");
        }
    }
    
    static void testPredefinedApproach(Scanner scanner) {
        System.out.println("=== Testing Predefined Java Methods ===");
        System.out.println("Choose a predefined method:");
        System.out.println("1. String.valueOf() method");
        System.out.println("2. Integer toString() method");
        System.out.println("3. Long toString() method");
        System.out.println("4. Character.isDigit() with Stream");
        System.out.println("5. Stream API with regex");
        System.out.println("6. DecimalFormat method");
        System.out.println("7. Test all predefined methods");
        System.out.print("Enter your choice (1-7): ");
        
        int methodChoice = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        
        switch(methodChoice) {
            case 1:
                System.out.print("Enter a number: ");
                String input1 = scanner.nextLine();
                try {
                    int count1 = countDigitsPredefined(input1);
                    System.out.println("Digits using String.valueOf(): " + count1);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid number format!");
                }
                break;
                
            case 2:
                System.out.print("Enter an integer: ");
                try {
                    int input2 = scanner.nextInt();
                    int count2 = countDigitsPredefinedInt(input2);
                    System.out.println("Digits using Integer toString(): " + count2);
                } catch (Exception e) {
                    System.out.println("Invalid integer!");
                }
                break;
                
            case 3:
                System.out.print("Enter a long number: ");
                try {
                    long input3 = scanner.nextLong();
                    int count3 = countDigitsPredefinedLong(input3);
                    System.out.println("Digits using Long toString(): " + count3);
                } catch (Exception e) {
                    System.out.println("Invalid long number!");
                }
                break;
                
            case 4:
                System.out.print("Enter any string with numbers: ");
                String input4 = scanner.nextLine();
                int count4 = countDigitsPredefinedAdvanced(input4);
                System.out.println("Total digits using Character.isDigit(): " + count4);
                break;
                
            case 5:
                System.out.print("Enter a number string: ");
                String input5 = scanner.nextLine();
                int count5 = countDigitsPredefinedStream(input5);
                System.out.println("Digits using Stream API: " + count5);
                break;
                
            case 6:
                System.out.print("Enter a decimal number: ");
                try {
                    double input6 = scanner.nextDouble();
                    int count6 = countDigitsPredefinedDecimalFormat(input6);
                    System.out.println("Digits using DecimalFormat: " + count6);
                } catch (Exception e) {
                    System.out.println("Invalid decimal number!");
                }
                break;
                
            case 7:
                testAllPredefinedMethods();
                break;
                
            default:
                System.out.println("Invalid choice!");
        }
    }
    
    static void testAllApproaches() {
        System.out.println("Testing all approaches with sample numbers:\n");
        
        // Test cases with different number ranges
        String[] testNumbers = {
            "0",
            "123",
            "-456",
            "1234567890",           // 10 digits
            "12345678901234567890", // 20 digits
            "999999999999999999999999999999", // 30 digits
            "-987654321098765432109876543210" // 31 digits (negative)
        };
        
        for (String testNum : testNumbers) {
            System.out.println("Testing number: " + testNum);
            
            // String approach
            int stringCount = countDigitsString(testNum);
            System.out.println("  String approach: " + stringCount + " digits");
            
            // Long approach (if number fits in long)
            try {
                long longNum = Long.parseLong(testNum);
                int longCount = countDigitsLong(longNum);
                int longLogCount = countDigitsLongLog(longNum);
                System.out.println("  Long approach: " + longCount + " digits");
                System.out.println("  Long (log) approach: " + longLogCount + " digits");
            } catch (NumberFormatException e) {
                System.out.println("  Long approach: Number too large for long");
            }
            
            // BigInteger approach
            try {
                BigInteger bigNum = new BigInteger(testNum);
                int bigCount = countDigitsBigInteger(bigNum);
                System.out.println("  BigInteger approach: " + bigCount + " digits");
            } catch (NumberFormatException e) {
                System.out.println("  BigInteger approach: Invalid format");
            }
            
            // Predefined methods testing
            try {
                if (testNum.matches("-?\\d+")) { // Valid integer string
                    System.out.println("  String.valueOf(): " + countDigitsPredefined(testNum));
                    System.out.println("  Character.isDigit(): " + countDigitsPredefinedAdvanced(testNum));
                    System.out.println("  Stream API: " + countDigitsPredefinedStream(testNum));
                } else {
                    System.out.println("  Predefined methods: Invalid format for conversion");
                }
            } catch (Exception e) {
                System.out.println("  Predefined methods: " + e.getMessage());
            }
            
            // Original int approach (if number fits in int)
            try {
                int intNum = Integer.parseInt(testNum);
                int intCount = countDigitOriginal(intNum);
                System.out.println("  Original (int) approach: " + intCount + " digits");
            } catch (NumberFormatException e) {
                System.out.println("  Original (int) approach: Number too large for int");
            }
            
            System.out.println();
        }
        
        // Performance comparison
        performanceComparison();
    }
    
    static void testAllPredefinedMethods() {
        System.out.println("=== Testing All Predefined Methods ===\n");
        
        // Test cases for predefined methods
        Object[] testCases = {
            123,
            -456,
            1234567890L,
            "789012345",
            "abc123def456",
            "00012300",
            123.456,
            -789.123,
            0,
            "0000"
        };
        
        for (Object testCase : testCases) {
            System.out.println("Testing with: " + testCase + " (" + testCase.getClass().getSimpleName() + ")");
            
            try {
                // Test different predefined methods based on input type
                if (testCase instanceof Integer) {
                    int intVal = (Integer) testCase;
                    System.out.println("  Integer toString(): " + countDigitsPredefinedInt(intVal));
                    System.out.println("  String.valueOf(): " + countDigitsPredefined(String.valueOf(intVal)));
                }
                
                if (testCase instanceof Long) {
                    long longVal = (Long) testCase;
                    System.out.println("  Long toString(): " + countDigitsPredefinedLong(longVal));
                    System.out.println("  String.valueOf(): " + countDigitsPredefined(String.valueOf(longVal)));
                }
                
                if (testCase instanceof String) {
                    String strVal = (String) testCase;
                    System.out.println("  Character.isDigit(): " + countDigitsPredefinedAdvanced(strVal));
                    System.out.println("  Stream API: " + countDigitsPredefinedStream(strVal));
                    
                    try {
                        System.out.println("  String.valueOf(): " + countDigitsPredefined(strVal));
                    } catch (NumberFormatException e) {
                        System.out.println("  String.valueOf(): Cannot parse as number");
                    }
                }
                
                if (testCase instanceof Double) {
                    double doubleVal = (Double) testCase;
                    System.out.println("  DecimalFormat: " + countDigitsPredefinedDecimalFormat(doubleVal));
                    System.out.println("  String.valueOf(): " + countDigitsPredefined(String.valueOf((long)doubleVal)));
                }
                
            } catch (Exception e) {
                System.out.println("  Error: " + e.getMessage());
            }
            
            System.out.println();
        }
        
        // Demonstrate unique features of each predefined method
        demonstratePredefinedFeatures();
    }
    
    static void demonstratePredefinedFeatures() {
        System.out.println("=== Unique Features of Each Predefined Method ===\n");
        
        // 1. String.valueOf() - Universal conversion
        System.out.println("1. String.valueOf() - Universal number conversion:");
        System.out.println("   Integer: " + countDigitsPredefinedInt(12345));
        System.out.println("   Long: " + countDigitsPredefinedLong(123456789012345L));
        System.out.println("   Feature: Handles all numeric types uniformly\n");
        
        // 2. Character.isDigit() - Filters only digits
        System.out.println("2. Character.isDigit() - Counts only digit characters:");
        System.out.println("   'abc123def456': " + countDigitsPredefinedAdvanced("abc123def456"));
        System.out.println("   '12.34.56': " + countDigitsPredefinedAdvanced("12.34.56"));
        System.out.println("   Feature: Ignores non-digit characters completely\n");
        
        // 3. Stream API - Modern Java approach
        System.out.println("3. Stream API - Modern functional approach:");
        System.out.println("   '000123000': " + countDigitsPredefinedStream("000123000"));
        System.out.println("   '007': " + countDigitsPredefinedStream("007"));
        System.out.println("   Feature: Handles leading zeros intelligently\n");
        
        // 4. DecimalFormat - Scientific notation
        System.out.println("4. DecimalFormat - Handles scientific notation:");
        System.out.println("   1.23E+5: " + countDigitsPredefinedDecimalFormat(1.23E+5));
        System.out.println("   9.87E-3: " + countDigitsPredefinedDecimalFormat(9.87E-3));
        System.out.println("   Feature: Converts scientific notation to decimal\n");
        
        // Performance comparison of predefined methods
        performancePredefinedMethods();
    }
    
    static void performancePredefinedMethods() {
        System.out.println("=== Performance Comparison: Predefined Methods ===");
        
        int iterations = 100000;
        long testLong = 1234567890123456L;
        String testString = "1234567890123456";
        double testDouble = 1234567890123456.0;
        
        // Test String.valueOf()
        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsPredefinedLong(testLong);
        }
        long stringValueOfTime = System.nanoTime() - startTime;
        
        // Test Character.isDigit()
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsPredefinedAdvanced(testString);
        }
        long charIsDigitTime = System.nanoTime() - startTime;
        
        // Test Stream API
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsPredefinedStream(testString);
        }
        long streamTime = System.nanoTime() - startTime;
        
        // Test DecimalFormat
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsPredefinedDecimalFormat(testDouble);
        }
        long decimalFormatTime = System.nanoTime() - startTime;
        
        System.out.println("Performance results (" + iterations + " iterations):");
        System.out.printf("String.valueOf(): %.2f ms\n", stringValueOfTime / 1_000_000.0);
        System.out.printf("Character.isDigit(): %.2f ms\n", charIsDigitTime / 1_000_000.0);
        System.out.printf("Stream API: %.2f ms\n", streamTime / 1_000_000.0);
        System.out.printf("DecimalFormat: %.2f ms\n", decimalFormatTime / 1_000_000.0);
        
        System.out.println("\nPredefined Methods Summary:");
        System.out.println("✓ String.valueOf(): Fastest, works with all numeric types");
        System.out.println("✓ Character.isDigit(): Best for mixed content (letters + numbers)");
        System.out.println("✓ Stream API: Most flexible, handles edge cases well");
        System.out.println("✓ DecimalFormat: Essential for scientific notation");
    }
    
    static void performanceComparison() {
        System.out.println("=== Performance Comparison ===");
        
        long testNumber = 1234567890123456789L;
        String testString = "1234567890123456789";
        BigInteger testBig = new BigInteger(testString);
        
        int iterations = 1000000;
        
        // Test long approach
        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsLong(testNumber);
        }
        long longTime = System.nanoTime() - startTime;
        
        // Test long logarithm approach
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsLongLog(testNumber);
        }
        long longLogTime = System.nanoTime() - startTime;
        
        // Test string approach
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsString(testString);
        }
        long stringTime = System.nanoTime() - startTime;
        
        // Test BigInteger approach
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsBigInteger(testBig);
        }
        long bigIntTime = System.nanoTime() - startTime;
        
        System.out.println("Performance results (" + iterations + " iterations):");
        System.out.printf("Long (division): %.2f ms\n", longTime / 1_000_000.0);
        System.out.printf("Long (logarithm): %.2f ms\n", longLogTime / 1_000_000.0);
        System.out.printf("String: %.2f ms\n", stringTime / 1_000_000.0);
        System.out.printf("BigInteger: %.2f ms\n", bigIntTime / 1_000_000.0);
        
        // Add predefined methods to performance comparison
        startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            countDigitsPredefinedLong(testNumber);
        }
        long predefinedTime = System.nanoTime() - startTime;
        System.out.printf("Predefined (String.valueOf): %.2f ms\n", predefinedTime / 1_000_000.0);
        
        System.out.println("\nRecommendations:");
        System.out.println("- For numbers up to 19 digits: Use long with logarithm method (fastest)");
        System.out.println("- For arbitrary length numbers: Use String method (good balance)");
        System.out.println("- For mathematical operations: Use BigInteger method");
        System.out.println("- For simple cases: Original int method is sufficient");
        System.out.println("- For predefined simplicity: Use String.valueOf() method (very fast & clean)");
        System.out.println("- For mixed content: Use Character.isDigit() with Stream API");
    }
}