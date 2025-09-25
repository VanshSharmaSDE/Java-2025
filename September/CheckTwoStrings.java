public class CheckTwoStrings{
    public static void main(String[] args) {
        String[] word1 = {"ab", "c"};
        String[] word2 = {"a", "bc"};
        // System.out.println(arrayStringsAreEqual(word1,word2));
        arrayStringsAreEqual(word1,word2);
    }

    static void arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder st1 = new StringBuilder();
        StringBuilder st2 = new StringBuilder();

        for(int i = 0 ; i < word1.length ; i++){
            st1.append(word1[i]);
        }

        for(int i = 0 ; i < word2.length ; i++){
            st2.append(word2[i]);
        }
        System.out.println(st1);
        System.out.println(st2);

        // Why this gives false:
        // StringBuilder.equals() uses Object's equals() method which compares references, not content
        // Even though st1 and st2 have same content "abc", they are different objects
        
        // WRONG WAY - compares object references, not content:
        if(st1.equals(st2)){
            System.out.println("StringBuilder.equals(): true");
        } else{
            System.out.println("StringBuilder.equals(): false");
        }
        
        // CORRECT WAY - convert to String first, then compare:
        if(st1.toString().equals(st2.toString())){
            System.out.println("toString().equals(): true");
        } else{
            System.out.println("toString().equals(): false");
        }

        // return false;
    }
}