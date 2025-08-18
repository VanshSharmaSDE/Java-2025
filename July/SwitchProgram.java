import java.util,*;

public class SwitchProgram {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a number (1-3): ");
        int choice = sc.nextInt();

        switch (choice) {
            case 1:
                System.out.println("You selected option 1.");
                break;
            case 2:
                System.out.println("You selected option 2.");
                break;
            case 3:
                System.out.println("You selected option 3.");
                break;
            default:
                System.out.println("Invalid option.");
        }


        System.out.println("Output from Enhanced Switch Statement:");
        // Enhanced Switch Statement
        switch (choice) {
            case 1:
            case 2:
            case 3:
                System.out.println("You selected option " + choice + ".");
                break;
            default:
                System.out.println("Invalid option.");
        }


        
        System.out.println("Output from New Switch Statement:");
        // New Switch Expression (Java 12+)
        String result = switch (choice) {
            case 1, 2, 3 -> "You selected option " + choice + ".";
            default -> "Invalid option.";
        };
        System.out.println(result);

        sc.close();
    }
}