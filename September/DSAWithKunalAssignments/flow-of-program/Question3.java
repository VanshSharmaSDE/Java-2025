public class Question3{
    public static void main(String[] args) {
        int num = 2;
        Table(num);
    }

    static void Table(int n){
        for (int i = 1; i <= 10; i++) {
            System.out.println(n*i);
        }
    }
}