import java.util.*;

public class ColoumnFirstIteration{
    public static void main(String[] args) {
        int[][] arr = {{1,2},
                       {2,3},
                       {3,4},
                       {4,5},
                       {5,6},
                       {6,7}};

    //    System.out.println(Arrays.toString(CF(arr)));
        CF(arr);
    }

    static void CF(int[][] arr){
        int rowLength = arr.length;
        int colLength = arr[0].length;
        for (int i = 0; i < colLength; i++) {
           for (int j = 0; j < rowLength; j++) {
               System.out.print(arr[j][i]);
           }
           System.out.println("\t");
        }
    }
}