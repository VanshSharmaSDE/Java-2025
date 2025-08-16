import java.util.*;

public class InputType {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a string: ");
        String str = sc.nextLine();
        System.out.print("Enter an integer: ");
        int num = sc.nextInt();
        System.out.print("Enter a double: ");
        double d = sc.nextDouble();

        System.out.println("You entered:");
        System.out.println("String: " + str);
        System.out.println("Integer: " + num);
        System.out.println("Double: " + d);

        // using Buffered Reader
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            System.out.print("Enter a string (BufferedReader): ");
            String str2 = br.readLine();
            System.out.print("Enter an integer (BufferedReader): ");
            int num2 = Integer.parseInt(br.readLine());
            System.out.print("Enter a double (BufferedReader): ");
            double d2 = Double.parseDouble(br.readLine());

            System.out.println("You entered (BufferedReader):");
            System.out.println("String: " + str2);
            System.out.println("Integer: " + num2);
            System.out.println("Double: " + d2);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // if you try to input like 1 2 3 = 123 means all the values are taken as a single string
        // and will not be parsed into their respective types
        String input = br.readLine();
        String[] tokens = input.split(" ");
        if (tokens.length == 3) {
            String str3 = tokens[0];
            int num3 = Integer.parseInt(tokens[1]);
            double d3 = Double.parseDouble(tokens[2]);

            System.out.println("You entered (BufferedReader with space-separated values):");
            System.out.println("String: " + str3);
            System.out.println("Integer: " + num3);
            System.out.println("Double: " + d3);
        } else {
            System.out.println("Invalid input format.");
        }

    }
}