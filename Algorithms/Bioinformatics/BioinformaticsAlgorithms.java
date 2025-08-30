package Algorithms.Bioinformatics;

import java.util.*;

/**
 * Comprehensive Bioinformatics Algorithms
 * DNA/RNA sequence analysis, protein folding, phylogenetics, and more
 */
public class BioinformaticsAlgorithms {
    
    /**
     * DNA/RNA Sequence Analysis
     */
    public static class SequenceAnalysis {
        
        /**
         * DNA to RNA transcription
         */
        public static String transcribe(String dna) {
            return dna.replace('T', 'U');
        }
        
        /**
         * RNA to Protein translation
         */
        public static String translate(String rna) {
            Map<String, Character> geneticCode = createGeneticCode();
            StringBuilder protein = new StringBuilder();
            
            for (int i = 0; i <= rna.length() - 3; i += 3) {
                String codon = rna.substring(i, i + 3);
                char aminoAcid = geneticCode.getOrDefault(codon, 'X');
                if (aminoAcid == '*') break; // Stop codon
                protein.append(aminoAcid);
            }
            
            return protein.toString();
        }
        
        private static Map<String, Character> createGeneticCode() {
            Map<String, Character> code = new HashMap<>();
            
            // Standard genetic code
            code.put("UUU", 'F'); code.put("UUC", 'F'); code.put("UUA", 'L'); code.put("UUG", 'L');
            code.put("UCU", 'S'); code.put("UCC", 'S'); code.put("UCA", 'S'); code.put("UCG", 'S');
            code.put("UAU", 'Y'); code.put("UAC", 'Y'); code.put("UAA", '*'); code.put("UAG", '*');
            code.put("UGU", 'C'); code.put("UGC", 'C'); code.put("UGA", '*'); code.put("UGG", 'W');
            
            code.put("CUU", 'L'); code.put("CUC", 'L'); code.put("CUA", 'L'); code.put("CUG", 'L');
            code.put("CCU", 'P'); code.put("CCC", 'P'); code.put("CCA", 'P'); code.put("CCG", 'P');
            code.put("CAU", 'H'); code.put("CAC", 'H'); code.put("CAA", 'Q'); code.put("CAG", 'Q');
            code.put("CGU", 'R'); code.put("CGC", 'R'); code.put("CGA", 'R'); code.put("CGG", 'R');
            
            code.put("AUU", 'I'); code.put("AUC", 'I'); code.put("AUA", 'I'); code.put("AUG", 'M');
            code.put("ACU", 'T'); code.put("ACC", 'T'); code.put("ACA", 'T'); code.put("ACG", 'T');
            code.put("AAU", 'N'); code.put("AAC", 'N'); code.put("AAA", 'K'); code.put("AAG", 'K');
            code.put("AGU", 'S'); code.put("AGC", 'S'); code.put("AGA", 'R'); code.put("AGG", 'R');
            
            code.put("GUU", 'V'); code.put("GUC", 'V'); code.put("GUA", 'V'); code.put("GUG", 'V');
            code.put("GCU", 'A'); code.put("GCC", 'A'); code.put("GCA", 'A'); code.put("GCG", 'A');
            code.put("GAU", 'D'); code.put("GAC", 'D'); code.put("GAA", 'E'); code.put("GAG", 'E');
            code.put("GGU", 'G'); code.put("GGC", 'G'); code.put("GGA", 'G'); code.put("GGG", 'G');
            
            return code;
        }
        
        /**
         * Reverse complement of DNA sequence
         */
        public static String reverseComplement(String dna) {
            StringBuilder result = new StringBuilder();
            Map<Character, Character> complement = Map.of('A', 'T', 'T', 'A', 'G', 'C', 'C', 'G');
            
            for (int i = dna.length() - 1; i >= 0; i--) {
                result.append(complement.get(dna.charAt(i)));
            }
            
            return result.toString();
        }
        
        /**
         * GC Content calculation
         */
        public static double gcContent(String sequence) {
            int gcCount = 0;
            for (char base : sequence.toCharArray()) {
                if (base == 'G' || base == 'C') {
                    gcCount++;
                }
            }
            return (double) gcCount / sequence.length() * 100;
        }
        
        /**
         * Find Open Reading Frames (ORFs)
         */
        public static List<String> findORFs(String dna) {
            List<String> orfs = new ArrayList<>();
            String rna = transcribe(dna);
            
            // Check all 6 reading frames (3 forward, 3 reverse)
            for (int frame = 0; frame < 3; frame++) {
                // Forward strand
                orfs.addAll(findORFsInFrame(rna, frame));
                
                // Reverse strand
                String reverseRna = transcribe(reverseComplement(dna));
                orfs.addAll(findORFsInFrame(reverseRna, frame));
            }
            
            return orfs;
        }
        
        private static List<String> findORFsInFrame(String rna, int frame) {
            List<String> orfs = new ArrayList<>();
            
            for (int i = frame; i <= rna.length() - 3; i += 3) {
                String codon = rna.substring(i, i + 3);
                
                if (codon.equals("AUG")) { // Start codon
                    StringBuilder orf = new StringBuilder();
                    
                    for (int j = i; j <= rna.length() - 3; j += 3) {
                        String currentCodon = rna.substring(j, j + 3);
                        orf.append(currentCodon);
                        
                        if (currentCodon.equals("UAA") || currentCodon.equals("UAG") || currentCodon.equals("UGA")) {
                            orfs.add(orf.toString());
                            break;
                        }
                    }
                }
            }
            
            return orfs;
        }
    }
    
    /**
     * Sequence Alignment Algorithms
     */
    public static class SequenceAlignment {
        
        /**
         * Needleman-Wunsch Global Alignment
         */
        public static AlignmentResult needlemanWunsch(String seq1, String seq2, int match, int mismatch, int gap) {
            int m = seq1.length();
            int n = seq2.length();
            
            // DP matrix
            int[][] dp = new int[m + 1][n + 1];
            
            // Initialize
            for (int i = 0; i <= m; i++) {
                dp[i][0] = i * gap;
            }
            for (int j = 0; j <= n; j++) {
                dp[0][j] = j * gap;
            }
            
            // Fill DP matrix
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    int scoreDiag = dp[i-1][j-1] + (seq1.charAt(i-1) == seq2.charAt(j-1) ? match : mismatch);
                    int scoreUp = dp[i-1][j] + gap;
                    int scoreLeft = dp[i][j-1] + gap;
                    
                    dp[i][j] = Math.max(Math.max(scoreDiag, scoreUp), scoreLeft);
                }
            }
            
            // Traceback
            StringBuilder align1 = new StringBuilder();
            StringBuilder align2 = new StringBuilder();
            
            int i = m, j = n;
            while (i > 0 || j > 0) {
                if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + (seq1.charAt(i-1) == seq2.charAt(j-1) ? match : mismatch)) {
                    align1.insert(0, seq1.charAt(i-1));
                    align2.insert(0, seq2.charAt(j-1));
                    i--; j--;
                } else if (i > 0 && dp[i][j] == dp[i-1][j] + gap) {
                    align1.insert(0, seq1.charAt(i-1));
                    align2.insert(0, '-');
                    i--;
                } else {
                    align1.insert(0, '-');
                    align2.insert(0, seq2.charAt(j-1));
                    j--;
                }
            }
            
            return new AlignmentResult(align1.toString(), align2.toString(), dp[m][n]);
        }
        
        /**
         * Smith-Waterman Local Alignment
         */
        public static AlignmentResult smithWaterman(String seq1, String seq2, int match, int mismatch, int gap) {
            int m = seq1.length();
            int n = seq2.length();
            
            // DP matrix
            int[][] dp = new int[m + 1][n + 1];
            int maxScore = 0;
            int maxI = 0, maxJ = 0;
            
            // Fill DP matrix
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    int scoreDiag = dp[i-1][j-1] + (seq1.charAt(i-1) == seq2.charAt(j-1) ? match : mismatch);
                    int scoreUp = dp[i-1][j] + gap;
                    int scoreLeft = dp[i][j-1] + gap;
                    
                    dp[i][j] = Math.max(0, Math.max(Math.max(scoreDiag, scoreUp), scoreLeft));
                    
                    if (dp[i][j] > maxScore) {
                        maxScore = dp[i][j];
                        maxI = i;
                        maxJ = j;
                    }
                }
            }
            
            // Traceback from maximum score
            StringBuilder align1 = new StringBuilder();
            StringBuilder align2 = new StringBuilder();
            
            int i = maxI, j = maxJ;
            while (i > 0 && j > 0 && dp[i][j] > 0) {
                if (dp[i][j] == dp[i-1][j-1] + (seq1.charAt(i-1) == seq2.charAt(j-1) ? match : mismatch)) {
                    align1.insert(0, seq1.charAt(i-1));
                    align2.insert(0, seq2.charAt(j-1));
                    i--; j--;
                } else if (dp[i][j] == dp[i-1][j] + gap) {
                    align1.insert(0, seq1.charAt(i-1));
                    align2.insert(0, '-');
                    i--;
                } else {
                    align1.insert(0, '-');
                    align2.insert(0, seq2.charAt(j-1));
                    j--;
                }
            }
            
            return new AlignmentResult(align1.toString(), align2.toString(), maxScore);
        }
        
        /**
         * Multiple Sequence Alignment using Progressive Alignment
         */
        public static List<String> progressiveAlignment(List<String> sequences) {
            if (sequences.size() < 2) return sequences;
            
            // Start with first two sequences
            AlignmentResult result = needlemanWunsch(sequences.get(0), sequences.get(1), 2, -1, -1);
            List<String> aligned = new ArrayList<>();
            aligned.add(result.sequence1);
            aligned.add(result.sequence2);
            
            // Add remaining sequences one by one
            for (int i = 2; i < sequences.size(); i++) {
                String consensus = buildConsensus(aligned);
                AlignmentResult newAlignment = needlemanWunsch(consensus, sequences.get(i), 2, -1, -1);
                
                // Update all existing alignments
                aligned = updateAlignments(aligned, newAlignment);
                aligned.add(newAlignment.sequence2);
            }
            
            return aligned;
        }
        
        private static String buildConsensus(List<String> sequences) {
            int length = sequences.get(0).length();
            StringBuilder consensus = new StringBuilder();
            
            for (int i = 0; i < length; i++) {
                Map<Character, Integer> counts = new HashMap<>();
                
                for (String seq : sequences) {
                    char c = seq.charAt(i);
                    counts.put(c, counts.getOrDefault(c, 0) + 1);
                }
                
                char mostFrequent = counts.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .get().getKey();
                
                consensus.append(mostFrequent);
            }
            
            return consensus.toString();
        }
        
        private static List<String> updateAlignments(List<String> sequences, AlignmentResult newAlignment) {
            // Simplified update - insert gaps where needed
            List<String> updated = new ArrayList<>();
            
            for (String seq : sequences) {
                updated.add(seq); // Simplified - would need proper gap insertion
            }
            
            return updated;
        }
        
        public static class AlignmentResult {
            public final String sequence1;
            public final String sequence2;
            public final int score;
            
            public AlignmentResult(String seq1, String seq2, int score) {
                this.sequence1 = seq1;
                this.sequence2 = seq2;
                this.score = score;
            }
            
            @Override
            public String toString() {
                return String.format("Score: %d\nSeq1: %s\nSeq2: %s", score, sequence1, sequence2);
            }
        }
    }
    
    /**
     * Phylogenetic Analysis
     */
    public static class PhylogeneticAnalysis {
        
        /**
         * UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
         */
        public static PhylogeneticTree upgma(double[][] distanceMatrix, String[] species) {
            int n = species.length;
            List<PhylogeneticNode> nodes = new ArrayList<>();
            
            // Initialize leaf nodes
            for (String name : species) {
                nodes.add(new PhylogeneticNode(name));
            }
            
            // Distance matrix (copy to avoid modification)
            double[][] distances = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(distanceMatrix[i], 0, distances[i], 0, n);
            }
            
            // Cluster until only one node remains
            while (nodes.size() > 1) {
                // Find minimum distance
                double minDist = Double.MAX_VALUE;
                int minI = 0, minJ = 1;
                
                for (int i = 0; i < nodes.size(); i++) {
                    for (int j = i + 1; j < nodes.size(); j++) {
                        if (distances[i][j] < minDist) {
                            minDist = distances[i][j];
                            minI = i;
                            minJ = j;
                        }
                    }
                }
                
                // Create new internal node
                PhylogeneticNode left = nodes.get(minI);
                PhylogeneticNode right = nodes.get(minJ);
                PhylogeneticNode parent = new PhylogeneticNode("Node" + (n - nodes.size() + 1));
                parent.left = left;
                parent.right = right;
                parent.height = minDist / 2;
                left.parent = parent;
                right.parent = parent;
                
                // Update distance matrix
                double[][] newDistances = new double[nodes.size() - 1][nodes.size() - 1];
                List<PhylogeneticNode> newNodes = new ArrayList<>();
                
                // Add new internal node
                newNodes.add(parent);
                
                // Add remaining nodes
                for (int i = 0; i < nodes.size(); i++) {
                    if (i != minI && i != minJ) {
                        newNodes.add(nodes.get(i));
                    }
                }
                
                // Calculate new distances
                for (int i = 1; i < newNodes.size(); i++) {
                    int originalI = findOriginalIndex(nodes, newNodes.get(i), minI, minJ);
                    newDistances[0][i] = newDistances[i][0] = 
                        (distances[minI][originalI] + distances[minJ][originalI]) / 2;
                }
                
                for (int i = 1; i < newNodes.size(); i++) {
                    for (int j = i + 1; j < newNodes.size(); j++) {
                        int originalI = findOriginalIndex(nodes, newNodes.get(i), minI, minJ);
                        int originalJ = findOriginalIndex(nodes, newNodes.get(j), minI, minJ);
                        newDistances[i][j] = newDistances[j][i] = distances[originalI][originalJ];
                    }
                }
                
                nodes = newNodes;
                distances = newDistances;
            }
            
            return new PhylogeneticTree(nodes.get(0));
        }
        
        private static int findOriginalIndex(List<PhylogeneticNode> originalNodes, 
                                           PhylogeneticNode node, int excludeI, int excludeJ) {
            for (int i = 0; i < originalNodes.size(); i++) {
                if (i != excludeI && i != excludeJ && originalNodes.get(i) == node) {
                    return i;
                }
            }
            return -1;
        }
        
        /**
         * Neighbor Joining Algorithm
         */
        public static PhylogeneticTree neighborJoining(double[][] distanceMatrix, String[] species) {
            int n = species.length;
            List<PhylogeneticNode> nodes = new ArrayList<>();
            
            // Initialize leaf nodes
            for (String name : species) {
                nodes.add(new PhylogeneticNode(name));
            }
            
            double[][] distances = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(distanceMatrix[i], 0, distances[i], 0, n);
            }
            
            while (nodes.size() > 2) {
                int size = nodes.size();
                
                // Calculate Q matrix
                double[][] Q = new double[size][size];
                double[] r = new double[size];
                
                // Calculate r values
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (i != j) {
                            r[i] += distances[i][j];
                        }
                    }
                }
                
                // Calculate Q matrix
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (i != j) {
                            Q[i][j] = (size - 2) * distances[i][j] - r[i] - r[j];
                        }
                    }
                }
                
                // Find minimum Q value
                double minQ = Double.MAX_VALUE;
                int minI = 0, minJ = 1;
                
                for (int i = 0; i < size; i++) {
                    for (int j = i + 1; j < size; j++) {
                        if (Q[i][j] < minQ) {
                            minQ = Q[i][j];
                            minI = i;
                            minJ = j;
                        }
                    }
                }
                
                // Create new internal node
                PhylogeneticNode left = nodes.get(minI);
                PhylogeneticNode right = nodes.get(minJ);
                PhylogeneticNode parent = new PhylogeneticNode("Node" + (n - nodes.size() + 1));
                parent.left = left;
                parent.right = right;
                
                // Calculate branch lengths
                double deltaIJ = distances[minI][minJ];
                double deltaIU = (deltaIJ + (r[minI] - r[minJ]) / (size - 2)) / 2;
                double deltaJU = deltaIJ - deltaIU;
                
                left.branchLength = deltaIU;
                right.branchLength = deltaJU;
                left.parent = parent;
                right.parent = parent;
                
                // Update nodes and distances (simplified)
                List<PhylogeneticNode> newNodes = new ArrayList<>();
                newNodes.add(parent);
                
                for (int i = 0; i < nodes.size(); i++) {
                    if (i != minI && i != minJ) {
                        newNodes.add(nodes.get(i));
                    }
                }
                
                nodes = newNodes;
                
                // Recalculate distance matrix (simplified)
                double[][] newDistances = new double[nodes.size()][nodes.size()];
                for (int i = 1; i < nodes.size(); i++) {
                    newDistances[0][i] = newDistances[i][0] = distances[minI][i] / 2 + distances[minJ][i] / 2;
                }
                
                for (int i = 1; i < nodes.size(); i++) {
                    for (int j = i + 1; j < nodes.size(); j++) {
                        newDistances[i][j] = newDistances[j][i] = distances[i][j];
                    }
                }
                
                distances = newDistances;
            }
            
            // Connect last two nodes
            if (nodes.size() == 2) {
                PhylogeneticNode root = new PhylogeneticNode("Root");
                root.left = nodes.get(0);
                root.right = nodes.get(1);
                nodes.get(0).parent = root;
                nodes.get(1).parent = root;
                return new PhylogeneticTree(root);
            }
            
            return new PhylogeneticTree(nodes.get(0));
        }
        
        public static class PhylogeneticNode {
            public String name;
            public PhylogeneticNode left, right, parent;
            public double height = 0;
            public double branchLength = 0;
            
            public PhylogeneticNode(String name) {
                this.name = name;
            }
            
            public boolean isLeaf() {
                return left == null && right == null;
            }
        }
        
        public static class PhylogeneticTree {
            public PhylogeneticNode root;
            
            public PhylogeneticTree(PhylogeneticNode root) {
                this.root = root;
            }
            
            public void printTree() {
                printTree(root, 0);
            }
            
            private void printTree(PhylogeneticNode node, int depth) {
                if (node != null) {
                    String indent = "  ".repeat(depth);
                    System.out.println(indent + node.name + 
                                     (node.isLeaf() ? " (leaf)" : " (internal)") +
                                     " height: " + node.height +
                                     " branch: " + node.branchLength);
                    
                    if (node.left != null) printTree(node.left, depth + 1);
                    if (node.right != null) printTree(node.right, depth + 1);
                }
            }
        }
    }
    
    /**
     * Protein Structure Analysis
     */
    public static class ProteinAnalysis {
        
        /**
         * Secondary structure prediction using Chou-Fasman method
         */
        public static String predictSecondaryStructure(String sequence) {
            // Simplified Chou-Fasman propensities
            Map<Character, Double> alphaPropensity = Map.of(
                'A', 1.42, 'E', 1.51, 'L', 1.21, 'M', 1.45,
                'D', 1.01, 'K', 1.16, 'R', 0.98, 'S', 0.77
            );
            
            Map<Character, Double> betaPropensity = Map.of(
                'V', 1.70, 'I', 1.60, 'F', 1.38, 'Y', 1.47,
                'C', 1.19, 'T', 1.19, 'W', 1.37, 'L', 1.30
            );
            
            StringBuilder structure = new StringBuilder();
            
            for (int i = 0; i < sequence.length(); i++) {
                char aa = sequence.charAt(i);
                double alphaP = alphaPropensity.getOrDefault(aa, 1.0);
                double betaP = betaPropensity.getOrDefault(aa, 1.0);
                
                if (alphaP > betaP && alphaP > 1.0) {
                    structure.append('H'); // Alpha helix
                } else if (betaP > alphaP && betaP > 1.0) {
                    structure.append('E'); // Beta sheet
                } else {
                    structure.append('C'); // Coil
                }
            }
            
            return structure.toString();
        }
        
        /**
         * Hydrophobicity analysis
         */
        public static double[] hydrophobicityProfile(String sequence) {
            // Kyte-Doolittle hydrophobicity scale
            Map<Character, Double> hydrophobicity = Map.of(
                'A', 1.8, 'R', -4.5, 'N', -3.5, 'D', -3.5, 'C', 2.5,
                'Q', -3.5, 'E', -3.5, 'G', -0.4, 'H', -3.2, 'I', 4.5,
                'L', 3.8, 'K', -3.9, 'M', 1.9, 'F', 2.8, 'P', -1.6,
                'S', -0.8, 'T', -0.7, 'W', -0.9, 'Y', -1.3, 'V', 4.2
            );
            
            int windowSize = 9;
            double[] profile = new double[sequence.length()];
            
            for (int i = 0; i < sequence.length(); i++) {
                double sum = 0;
                int count = 0;
                
                for (int j = Math.max(0, i - windowSize/2); 
                     j <= Math.min(sequence.length() - 1, i + windowSize/2); j++) {
                    sum += hydrophobicity.getOrDefault(sequence.charAt(j), 0.0);
                    count++;
                }
                
                profile[i] = sum / count;
            }
            
            return profile;
        }
        
        /**
         * Transmembrane domain prediction
         */
        public static List<int[]> predictTransmembraneDomains(String sequence) {
            double[] hydrophobicity = hydrophobicityProfile(sequence);
            List<int[]> domains = new ArrayList<>();
            
            boolean inDomain = false;
            int domainStart = 0;
            double threshold = 1.6; // Hydrophobicity threshold
            int minLength = 15; // Minimum transmembrane domain length
            
            for (int i = 0; i < hydrophobicity.length; i++) {
                if (!inDomain && hydrophobicity[i] > threshold) {
                    inDomain = true;
                    domainStart = i;
                } else if (inDomain && hydrophobicity[i] <= threshold) {
                    if (i - domainStart >= minLength) {
                        domains.add(new int[]{domainStart, i - 1});
                    }
                    inDomain = false;
                }
            }
            
            // Check last domain
            if (inDomain && hydrophobicity.length - domainStart >= minLength) {
                domains.add(new int[]{domainStart, hydrophobicity.length - 1});
            }
            
            return domains;
        }
    }
    
    /**
     * Genetic Algorithm for Molecular Evolution
     */
    public static class MolecularEvolution {
        
        /**
         * Simulate molecular evolution using genetic algorithm
         */
        public static List<String> simulateEvolution(String ancestralSequence, int generations, 
                                                    double mutationRate, int populationSize) {
            List<String> population = new ArrayList<>();
            
            // Initialize population
            for (int i = 0; i < populationSize; i++) {
                population.add(ancestralSequence);
            }
            
            Random random = new Random();
            char[] bases = {'A', 'T', 'G', 'C'};
            
            for (int gen = 0; gen < generations; gen++) {
                List<String> newPopulation = new ArrayList<>();
                
                for (String individual : population) {
                    StringBuilder mutated = new StringBuilder(individual);
                    
                    // Apply mutations
                    for (int i = 0; i < mutated.length(); i++) {
                        if (random.nextDouble() < mutationRate) {
                            mutated.setCharAt(i, bases[random.nextInt(4)]);
                        }
                    }
                    
                    newPopulation.add(mutated.toString());
                }
                
                // Selection (simplified - random selection)
                Collections.shuffle(newPopulation);
                population = newPopulation.subList(0, populationSize);
            }
            
            return population;
        }
        
        /**
         * Calculate evolutionary distance
         */
        public static double evolutionaryDistance(String seq1, String seq2) {
            if (seq1.length() != seq2.length()) {
                throw new IllegalArgumentException("Sequences must have same length");
            }
            
            int differences = 0;
            for (int i = 0; i < seq1.length(); i++) {
                if (seq1.charAt(i) != seq2.charAt(i)) {
                    differences++;
                }
            }
            
            return (double) differences / seq1.length();
        }
        
        /**
         * Jukes-Cantor distance correction
         */
        public static double jukesCantorDistance(double p) {
            if (p >= 0.75) {
                return Double.POSITIVE_INFINITY;
            }
            return -0.75 * Math.log(1 - (4.0/3.0) * p);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Bioinformatics Algorithms Demo:");
        System.out.println("===============================");
        
        // Sequence Analysis
        System.out.println("1. DNA/RNA Analysis:");
        String dna = "ATGGCATAA";
        String rna = SequenceAnalysis.transcribe(dna);
        String protein = SequenceAnalysis.translate(rna);
        
        System.out.println("DNA: " + dna);
        System.out.println("RNA: " + rna);
        System.out.println("Protein: " + protein);
        System.out.println("GC Content: " + SequenceAnalysis.gcContent(dna) + "%");
        System.out.println("Reverse Complement: " + SequenceAnalysis.reverseComplement(dna));
        
        // Sequence Alignment
        System.out.println("\n2. Sequence Alignment:");
        String seq1 = "GCATGCU";
        String seq2 = "GATTACA";
        
        SequenceAlignment.AlignmentResult globalAlign = 
            SequenceAlignment.needlemanWunsch(seq1, seq2, 2, -1, -1);
        System.out.println("Global Alignment:");
        System.out.println(globalAlign);
        
        SequenceAlignment.AlignmentResult localAlign = 
            SequenceAlignment.smithWaterman(seq1, seq2, 2, -1, -1);
        System.out.println("\nLocal Alignment:");
        System.out.println(localAlign);
        
        // Phylogenetic Analysis
        System.out.println("\n3. Phylogenetic Analysis:");
        String[] species = {"Human", "Chimp", "Gorilla", "Orangutan"};
        double[][] distanceMatrix = {
            {0.0, 0.015, 0.020, 0.025},
            {0.015, 0.0, 0.022, 0.027},
            {0.020, 0.022, 0.0, 0.030},
            {0.025, 0.027, 0.030, 0.0}
        };
        
        PhylogeneticAnalysis.PhylogeneticTree tree = 
            PhylogeneticAnalysis.upgma(distanceMatrix, species);
        System.out.println("UPGMA Tree:");
        tree.printTree();
        
        // Protein Analysis
        System.out.println("\n4. Protein Structure Analysis:");
        String proteinSeq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN";
        String secondaryStructure = ProteinAnalysis.predictSecondaryStructure(proteinSeq.substring(0, 50));
        System.out.println("Sequence: " + proteinSeq.substring(0, 50));
        System.out.println("Structure: " + secondaryStructure);
        
        double[] hydrophobicity = ProteinAnalysis.hydrophobicityProfile(proteinSeq.substring(0, 50));
        System.out.println("Average hydrophobicity: " + 
                          Arrays.stream(hydrophobicity).average().orElse(0.0));
        
        // Molecular Evolution
        System.out.println("\n5. Molecular Evolution:");
        List<String> evolvedSequences = MolecularEvolution.simulateEvolution(
            "ATGCATGCATGC", 100, 0.01, 10);
        
        System.out.println("Original: ATGCATGCATGC");
        System.out.println("Evolved samples:");
        for (int i = 0; i < Math.min(5, evolvedSequences.size()); i++) {
            String evolved = evolvedSequences.get(i);
            double distance = MolecularEvolution.evolutionaryDistance("ATGCATGCATGC", evolved);
            System.out.println(evolved + " (distance: " + String.format("%.3f", distance) + ")");
        }
        
        // ORF Finding
        System.out.println("\n6. Open Reading Frame Analysis:");
        String testDNA = "ATGAAATTTAAATAG";
        List<String> orfs = SequenceAnalysis.findORFs(testDNA);
        System.out.println("DNA: " + testDNA);
        System.out.println("ORFs found: " + orfs.size());
        for (String orf : orfs) {
            System.out.println("ORF: " + orf);
        }
    }
}
