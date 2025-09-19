import java.util.Scanner;

public class CalculateCGPA {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Enter number of subjects: ");
        int numSubjects = sc.nextInt();
        
        double totalGradePoints = 0;
        int totalCredits = 0;
        
        for (int i = 1; i <= numSubjects; i++) {
            System.out.print("Enter grade points for subject " + i + ": ");
            double gradePoints = sc.nextDouble();
            
            System.out.print("Enter credits for subject " + i + ": ");
            int credits = sc.nextInt();
            
            totalGradePoints += gradePoints * credits;
            totalCredits += credits;
        }
        
        if (totalCredits == 0) {
            System.out.println("No credits entered!");
        } else {
            double cgpa = totalGradePoints / totalCredits;
            System.out.println("CGPA: " + String.format("%.2f", cgpa));
        }
        
        sc.close();
    }
}