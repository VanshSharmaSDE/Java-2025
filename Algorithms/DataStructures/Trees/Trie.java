package Algorithms.DataStructures.Trees;

/**
 * Trie (Prefix Tree) Implementation
 * Efficient for string operations like autocomplete and spell checking
 */
public class Trie {
    
    private TrieNode root;
    private int size;
    
    private static class TrieNode {
        TrieNode[] children;
        boolean isEndOfWord;
        int frequency; // For word frequency counting
        
        TrieNode() {
            children = new TrieNode[26]; // For lowercase a-z
            isEndOfWord = false;
            frequency = 0;
        }
    }
    
    public Trie() {
        root = new TrieNode();
        size = 0;
    }
    
    /**
     * Insert word into trie
     * Time Complexity: O(m) where m is length of word
     */
    public void insert(String word) {
        if (word == null || word.isEmpty()) return;
        
        TrieNode current = root;
        word = word.toLowerCase();
        
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (index < 0 || index >= 26) continue; // Skip invalid characters
            
            if (current.children[index] == null) {
                current.children[index] = new TrieNode();
            }
            current = current.children[index];
        }
        
        if (!current.isEndOfWord) {
            size++;
        }
        current.isEndOfWord = true;
        current.frequency++;
    }
    
    /**
     * Search for word in trie
     * Time Complexity: O(m) where m is length of word
     */
    public boolean search(String word) {
        TrieNode node = searchNode(word);
        return node != null && node.isEndOfWord;
    }
    
    /**
     * Check if any word starts with given prefix
     * Time Complexity: O(m) where m is length of prefix
     */
    public boolean startsWith(String prefix) {
        return searchNode(prefix) != null;
    }
    
    private TrieNode searchNode(String str) {
        if (str == null) return null;
        
        TrieNode current = root;
        str = str.toLowerCase();
        
        for (char c : str.toCharArray()) {
            int index = c - 'a';
            if (index < 0 || index >= 26 || current.children[index] == null) {
                return null;
            }
            current = current.children[index];
        }
        
        return current;
    }
    
    /**
     * Delete word from trie
     * Time Complexity: O(m) where m is length of word
     */
    public void delete(String word) {
        if (word == null || word.isEmpty()) return;
        delete(root, word.toLowerCase(), 0);
    }
    
    private boolean delete(TrieNode current, String word, int index) {
        if (index == word.length()) {
            if (!current.isEndOfWord) {
                return false; // Word doesn't exist
            }
            current.isEndOfWord = false;
            current.frequency = 0;
            size--;
            
            // Return true if current has no children (can be deleted)
            return !hasChildren(current);
        }
        
        char c = word.charAt(index);
        int charIndex = c - 'a';
        TrieNode node = current.children[charIndex];
        
        if (node == null) {
            return false; // Word doesn't exist
        }
        
        boolean shouldDeleteChild = delete(node, word, index + 1);
        
        if (shouldDeleteChild) {
            current.children[charIndex] = null;
            
            // Return true if current has no children and is not end of another word
            return !current.isEndOfWord && !hasChildren(current);
        }
        
        return false;
    }
    
    private boolean hasChildren(TrieNode node) {
        for (TrieNode child : node.children) {
            if (child != null) return true;
        }
        return false;
    }
    
    /**
     * Get all words with given prefix
     * Time Complexity: O(p + n) where p is prefix length, n is number of words with prefix
     */
    public java.util.List<String> getWordsWithPrefix(String prefix) {
        java.util.List<String> result = new java.util.ArrayList<>();
        TrieNode prefixNode = searchNode(prefix);
        
        if (prefixNode != null) {
            getAllWords(prefixNode, prefix.toLowerCase(), result);
        }
        
        return result;
    }
    
    private void getAllWords(TrieNode node, String currentWord, java.util.List<String> result) {
        if (node.isEndOfWord) {
            result.add(currentWord);
        }
        
        for (int i = 0; i < 26; i++) {
            if (node.children[i] != null) {
                getAllWords(node.children[i], currentWord + (char)(i + 'a'), result);
            }
        }
    }
    
    /**
     * Get all words in trie
     * Time Complexity: O(n) where n is total number of characters in all words
     */
    public java.util.List<String> getAllWords() {
        java.util.List<String> result = new java.util.ArrayList<>();
        getAllWords(root, "", result);
        return result;
    }
    
    /**
     * Auto-complete suggestions
     * Returns up to limit number of words starting with prefix
     */
    public java.util.List<String> autoComplete(String prefix, int limit) {
        java.util.List<String> suggestions = getWordsWithPrefix(prefix);
        
        // Sort by frequency (descending) then alphabetically
        suggestions.sort((a, b) -> {
            TrieNode nodeA = searchNode(a);
            TrieNode nodeB = searchNode(b);
            int freqCompare = Integer.compare(nodeB.frequency, nodeA.frequency);
            return freqCompare != 0 ? freqCompare : a.compareTo(b);
        });
        
        return suggestions.subList(0, Math.min(limit, suggestions.size()));
    }
    
    /**
     * Spell checking with suggestions
     * Returns words within edit distance of 1
     */
    public java.util.List<String> spellCheck(String word) {
        java.util.List<String> suggestions = new java.util.ArrayList<>();
        
        if (search(word)) {
            suggestions.add(word); // Word is correctly spelled
            return suggestions;
        }
        
        // Generate words with edit distance 1
        java.util.Set<String> candidates = new java.util.HashSet<>();
        
        // Deletions
        for (int i = 0; i < word.length(); i++) {
            String candidate = word.substring(0, i) + word.substring(i + 1);
            candidates.add(candidate);
        }
        
        // Insertions
        for (int i = 0; i <= word.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                String candidate = word.substring(0, i) + c + word.substring(i);
                candidates.add(candidate);
            }
        }
        
        // Substitutions
        for (int i = 0; i < word.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                String candidate = word.substring(0, i) + c + word.substring(i + 1);
                candidates.add(candidate);
            }
        }
        
        // Transpositions
        for (int i = 0; i < word.length() - 1; i++) {
            String candidate = word.substring(0, i) + word.charAt(i + 1) + 
                             word.charAt(i) + word.substring(i + 2);
            candidates.add(candidate);
        }
        
        // Check which candidates exist in trie
        for (String candidate : candidates) {
            if (search(candidate)) {
                suggestions.add(candidate);
            }
        }
        
        return suggestions;
    }
    
    /**
     * Find longest common prefix of all words
     */
    public String longestCommonPrefix() {
        if (size == 0) return "";
        
        StringBuilder lcp = new StringBuilder();
        TrieNode current = root;
        
        while (countChildren(current) == 1 && !current.isEndOfWord) {
            for (int i = 0; i < 26; i++) {
                if (current.children[i] != null) {
                    lcp.append((char)(i + 'a'));
                    current = current.children[i];
                    break;
                }
            }
        }
        
        return lcp.toString();
    }
    
    private int countChildren(TrieNode node) {
        int count = 0;
        for (TrieNode child : node.children) {
            if (child != null) count++;
        }
        return count;
    }
    
    /**
     * Count words with given prefix
     */
    public int countWordsWithPrefix(String prefix) {
        TrieNode prefixNode = searchNode(prefix);
        if (prefixNode == null) return 0;
        
        return countWords(prefixNode);
    }
    
    private int countWords(TrieNode node) {
        int count = node.isEndOfWord ? 1 : 0;
        
        for (TrieNode child : node.children) {
            if (child != null) {
                count += countWords(child);
            }
        }
        
        return count;
    }
    
    /**
     * Get word frequency
     */
    public int getWordFrequency(String word) {
        TrieNode node = searchNode(word);
        return (node != null && node.isEndOfWord) ? node.frequency : 0;
    }
    
    public int size() {
        return size;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Display trie statistics
     */
    public void displayStats() {
        System.out.println("Trie Statistics:");
        System.out.println("Total words: " + size);
        System.out.println("Longest common prefix: \"" + longestCommonPrefix() + "\"");
        
        if (!isEmpty()) {
            java.util.List<String> allWords = getAllWords();
            System.out.println("Sample words: " + 
                allWords.subList(0, Math.min(10, allWords.size())));
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Trie (Prefix Tree) Demo:");
        System.out.println("========================");
        
        Trie trie = new Trie();
        
        // Insert words
        System.out.println("Inserting words: cat, car, card, care, careful, cars, carry");
        String[] words = {"cat", "car", "card", "care", "careful", "cars", "carry"};
        for (String word : words) {
            trie.insert(word);
        }
        
        trie.displayStats();
        
        // Search operations
        System.out.println("\nSearch operations:");
        System.out.println("Search 'car': " + trie.search("car"));
        System.out.println("Search 'care': " + trie.search("care"));
        System.out.println("Search 'caring': " + trie.search("caring"));
        
        // Prefix operations
        System.out.println("\nPrefix operations:");
        System.out.println("Starts with 'car': " + trie.startsWith("car"));
        System.out.println("Words with prefix 'car': " + trie.getWordsWithPrefix("car"));
        System.out.println("Count words with prefix 'car': " + trie.countWordsWithPrefix("car"));
        
        // Auto-complete
        System.out.println("\nAuto-complete for 'ca' (limit 5):");
        System.out.println(trie.autoComplete("ca", 5));
        
        // Add word frequencies
        trie.insert("car"); // Increase frequency
        trie.insert("car");
        trie.insert("care");
        
        System.out.println("\nWord frequencies:");
        System.out.println("'car' frequency: " + trie.getWordFrequency("car"));
        System.out.println("'care' frequency: " + trie.getWordFrequency("care"));
        
        // Spell checking
        System.out.println("\nSpell checking:");
        System.out.println("Suggestions for 'carr': " + trie.spellCheck("carr"));
        System.out.println("Suggestions for 'cra': " + trie.spellCheck("cra"));
        
        // Delete operations
        System.out.println("\nDeleting 'cars':");
        trie.delete("cars");
        System.out.println("Search 'cars': " + trie.search("cars"));
        System.out.println("Words with prefix 'car': " + trie.getWordsWithPrefix("car"));
        
        // Performance test
        System.out.println("\nPerformance Test - Inserting 1000 words:");
        Trie perfTrie = new Trie();
        long startTime = System.nanoTime();
        
        for (int i = 0; i < 1000; i++) {
            perfTrie.insert("word" + i);
        }
        
        long endTime = System.nanoTime();
        System.out.println("Time taken: " + (endTime - startTime) / 1_000_000.0 + " ms");
        System.out.println("Size: " + perfTrie.size());
        
        // Test prefix search performance
        startTime = System.nanoTime();
        java.util.List<String> prefixResults = perfTrie.getWordsWithPrefix("word1");
        endTime = System.nanoTime();
        System.out.println("Prefix search time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        System.out.println("Results found: " + prefixResults.size());
    }
}
