public class Question1{
    public static void main(String[] args) {
        int year = 1997;
        System.out.println(LeapYear(year));
    }

    static String LeapYear(int n){
        if(n % 4 == 0) return "LeapYear";
        return "Not a Leap Year";
    }
}