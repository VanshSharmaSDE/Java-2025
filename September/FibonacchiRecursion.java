public class FibonacchiRecursion{
    public static void main(String[] args) {
        System.out.println(FB(40));
    }

    static int FB(int n){

        if(n == 0){
            return 0;
        }

        if(n == 1){
            return 1;
        }

        return FB(n - 1) + FB(n - 2);
    }
}