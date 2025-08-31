package Algorithms.ArtificialIntelligence;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * Comprehensive Artificial Intelligence Algorithms
 * Search algorithms, expert systems, planning, and reasoning
 */
public class AIAlgorithms {
    
    /**
     * A* Search Algorithm with Heuristics
     */
    public static class AStarSearch {
        
        public static class Node implements Comparable<Node> {
            public final int x, y;
            public final double gCost; // Cost from start
            public final double hCost; // Heuristic cost to goal
            public final double fCost; // Total cost
            public final Node parent;
            
            public Node(int x, int y, double gCost, double hCost, Node parent) {
                this.x = x;
                this.y = y;
                this.gCost = gCost;
                this.hCost = hCost;
                this.fCost = gCost + hCost;
                this.parent = parent;
            }
            
            @Override
            public int compareTo(Node other) {
                return Double.compare(this.fCost, other.fCost);
            }
            
            @Override
            public boolean equals(Object obj) {
                if (!(obj instanceof Node)) return false;
                Node other = (Node) obj;
                return x == other.x && y == other.y;
            }
            
            @Override
            public int hashCode() {
                return Objects.hash(x, y);
            }
            
            public String toString() {
                return String.format("(%d,%d)[g=%.1f,h=%.1f,f=%.1f]", x, y, gCost, hCost, fCost);
            }
        }
        
        public static class Grid {
            private final int[][] grid;
            private final int width, height;
            
            public Grid(int[][] grid) {
                this.grid = grid;
                this.height = grid.length;
                this.width = grid[0].length;
            }
            
            public boolean isWalkable(int x, int y) {
                return x >= 0 && x < width && y >= 0 && y < height && grid[y][x] == 0;
            }
            
            public List<Node> getNeighbors(Node node) {
                List<Node> neighbors = new ArrayList<>();
                int[][] directions = {{0,1}, {1,0}, {0,-1}, {-1,0}, {1,1}, {-1,1}, {1,-1}, {-1,-1}};
                
                for (int[] dir : directions) {
                    int newX = node.x + dir[0];
                    int newY = node.y + dir[1];
                    
                    if (isWalkable(newX, newY)) {
                        double moveCost = (dir[0] != 0 && dir[1] != 0) ? 1.414 : 1.0; // Diagonal vs straight
                        neighbors.add(new Node(newX, newY, node.gCost + moveCost, 0, node));
                    }
                }
                
                return neighbors;
            }
            
            public double manhattanDistance(int x1, int y1, int x2, int y2) {
                return Math.abs(x1 - x2) + Math.abs(y1 - y2);
            }
            
            public double euclideanDistance(int x1, int y1, int x2, int y2) {
                return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
            }
        }
        
        public static List<Node> findPath(Grid grid, int startX, int startY, int goalX, int goalY) {
            PriorityQueue<Node> openSet = new PriorityQueue<>();
            Set<Node> closedSet = new HashSet<>();
            Map<String, Node> allNodes = new HashMap<>();
            
            Node startNode = new Node(startX, startY, 0, 
                                    grid.euclideanDistance(startX, startY, goalX, goalY), null);
            openSet.add(startNode);
            allNodes.put(startX + "," + startY, startNode);
            
            while (!openSet.isEmpty()) {
                Node current = openSet.poll();
                closedSet.add(current);
                
                // Goal reached
                if (current.x == goalX && current.y == goalY) {
                    return reconstructPath(current);
                }
                
                for (Node neighbor : grid.getNeighbors(current)) {
                    if (closedSet.contains(neighbor)) continue;
                    
                    neighbor = new Node(neighbor.x, neighbor.y, neighbor.gCost,
                                      grid.euclideanDistance(neighbor.x, neighbor.y, goalX, goalY),
                                      current);
                    
                    String key = neighbor.x + "," + neighbor.y;
                    Node existingNode = allNodes.get(key);
                    
                    if (existingNode == null || neighbor.gCost < existingNode.gCost) {
                        allNodes.put(key, neighbor);
                        openSet.add(neighbor);
                    }
                }
            }
            
            return new ArrayList<>(); // No path found
        }
        
        private static List<Node> reconstructPath(Node goalNode) {
            List<Node> path = new ArrayList<>();
            Node current = goalNode;
            
            while (current != null) {
                path.add(current);
                current = current.parent;
            }
            
            Collections.reverse(path);
            return path;
        }
    }
    
    /**
     * Minimax Algorithm with Alpha-Beta Pruning
     */
    public static class MinimaxAlgorithm {
        
        public static class GameState {
            private final int[][] board;
            private final int currentPlayer;
            private final int boardSize;
            
            public GameState(int[][] board, int currentPlayer) {
                this.boardSize = board.length;
                this.board = new int[boardSize][boardSize];
                for (int i = 0; i < boardSize; i++) {
                    System.arraycopy(board[i], 0, this.board[i], 0, boardSize);
                }
                this.currentPlayer = currentPlayer;
            }
            
            public List<GameState> getSuccessors() {
                List<GameState> successors = new ArrayList<>();
                
                for (int i = 0; i < boardSize; i++) {
                    for (int j = 0; j < boardSize; j++) {
                        if (board[i][j] == 0) {
                            int[][] newBoard = copyBoard();
                            newBoard[i][j] = currentPlayer;
                            successors.add(new GameState(newBoard, -currentPlayer));
                        }
                    }
                }
                
                return successors;
            }
            
            private int[][] copyBoard() {
                int[][] copy = new int[boardSize][boardSize];
                for (int i = 0; i < boardSize; i++) {
                    System.arraycopy(board[i], 0, copy[i], 0, boardSize);
                }
                return copy;
            }
            
            public boolean isTerminal() {
                return isWin(1) || isWin(-1) || isFull();
            }
            
            public boolean isWin(int player) {
                // Check rows, columns, and diagonals
                for (int i = 0; i < boardSize; i++) {
                    if (checkLine(board[i], player)) return true; // Row
                    int[] column = new int[boardSize];
                    for (int j = 0; j < boardSize; j++) {
                        column[j] = board[j][i];
                    }
                    if (checkLine(column, player)) return true; // Column
                }
                
                // Diagonals
                int[] diag1 = new int[boardSize];
                int[] diag2 = new int[boardSize];
                for (int i = 0; i < boardSize; i++) {
                    diag1[i] = board[i][i];
                    diag2[i] = board[i][boardSize - 1 - i];
                }
                
                return checkLine(diag1, player) || checkLine(diag2, player);
            }
            
            private boolean checkLine(int[] line, int player) {
                for (int cell : line) {
                    if (cell != player) return false;
                }
                return true;
            }
            
            public boolean isFull() {
                for (int[] row : board) {
                    for (int cell : row) {
                        if (cell == 0) return false;
                    }
                }
                return true;
            }
            
            public int evaluate() {
                if (isWin(1)) return 100;
                if (isWin(-1)) return -100;
                
                // Evaluate based on potential wins
                int score = 0;
                score += evaluateLines(1) - evaluateLines(-1);
                return score;
            }
            
            private int evaluateLines(int player) {
                int score = 0;
                
                // Evaluate all possible lines
                for (int i = 0; i < boardSize; i++) {
                    score += evaluateLine(board[i], player);
                    int[] column = new int[boardSize];
                    for (int j = 0; j < boardSize; j++) {
                        column[j] = board[j][i];
                    }
                    score += evaluateLine(column, player);
                }
                
                // Diagonals
                int[] diag1 = new int[boardSize];
                int[] diag2 = new int[boardSize];
                for (int i = 0; i < boardSize; i++) {
                    diag1[i] = board[i][i];
                    diag2[i] = board[i][boardSize - 1 - i];
                }
                score += evaluateLine(diag1, player);
                score += evaluateLine(diag2, player);
                
                return score;
            }
            
            private int evaluateLine(int[] line, int player) {
                int playerCount = 0;
                int opponentCount = 0;
                
                for (int cell : line) {
                    if (cell == player) playerCount++;
                    else if (cell == -player) opponentCount++;
                }
                
                if (opponentCount > 0 && playerCount > 0) return 0; // Blocked line
                if (playerCount == boardSize) return 100;
                if (opponentCount == boardSize) return -100;
                
                return playerCount * playerCount; // Exponential scoring
            }
            
            public int getCurrentPlayer() { return currentPlayer; }
            public int[][] getBoard() { return board; }
        }
        
        public static class MinimaxResult {
            public final int score;
            public final GameState bestMove;
            public final int nodesExplored;
            
            public MinimaxResult(int score, GameState bestMove, int nodesExplored) {
                this.score = score;
                this.bestMove = bestMove;
                this.nodesExplored = nodesExplored;
            }
        }
        
        private static int nodesExplored;
        
        public static MinimaxResult minimax(GameState state, int depth, boolean maximizing) {
            nodesExplored = 0;
            int score = minimaxRecursive(state, depth, maximizing, Integer.MIN_VALUE, Integer.MAX_VALUE);
            return new MinimaxResult(score, findBestMove(state, depth, maximizing), nodesExplored);
        }
        
        private static int minimaxRecursive(GameState state, int depth, boolean maximizing, int alpha, int beta) {
            nodesExplored++;
            
            if (depth == 0 || state.isTerminal()) {
                return state.evaluate();
            }
            
            if (maximizing) {
                int maxEval = Integer.MIN_VALUE;
                for (GameState child : state.getSuccessors()) {
                    int eval = minimaxRecursive(child, depth - 1, false, alpha, beta);
                    maxEval = Math.max(maxEval, eval);
                    alpha = Math.max(alpha, eval);
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
                return maxEval;
            } else {
                int minEval = Integer.MAX_VALUE;
                for (GameState child : state.getSuccessors()) {
                    int eval = minimaxRecursive(child, depth - 1, true, alpha, beta);
                    minEval = Math.min(minEval, eval);
                    beta = Math.min(beta, eval);
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
                return minEval;
            }
        }
        
        private static GameState findBestMove(GameState state, int depth, boolean maximizing) {
            GameState bestMove = null;
            int bestScore = maximizing ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            
            for (GameState child : state.getSuccessors()) {
                int score = minimaxRecursive(child, depth - 1, !maximizing, Integer.MIN_VALUE, Integer.MAX_VALUE);
                
                if ((maximizing && score > bestScore) || (!maximizing && score < bestScore)) {
                    bestScore = score;
                    bestMove = child;
                }
            }
            
            return bestMove;
        }
    }
    
    /**
     * Genetic Algorithm Implementation
     */
    public static class GeneticAlgorithm {
        
        public static class Individual {
            private final double[] genes;
            private double fitness;
            
            public Individual(int geneLength) {
                genes = new double[geneLength];
                Random random = new Random();
                for (int i = 0; i < geneLength; i++) {
                    genes[i] = random.nextGaussian();
                }
                fitness = 0;
            }
            
            public Individual(double[] genes) {
                this.genes = genes.clone();
                fitness = 0;
            }
            
            public void calculateFitness(java.util.function.Function<double[], Double> fitnessFunction) {
                fitness = fitnessFunction.apply(genes);
            }
            
            public Individual crossover(Individual partner, double crossoverRate) {
                Random random = new Random();
                double[] childGenes = new double[genes.length];
                
                for (int i = 0; i < genes.length; i++) {
                    if (random.nextDouble() < crossoverRate) {
                        childGenes[i] = partner.genes[i];
                    } else {
                        childGenes[i] = genes[i];
                    }
                }
                
                return new Individual(childGenes);
            }
            
            public void mutate(double mutationRate, double mutationStrength) {
                Random random = new Random();
                for (int i = 0; i < genes.length; i++) {
                    if (random.nextDouble() < mutationRate) {
                        genes[i] += random.nextGaussian() * mutationStrength;
                    }
                }
            }
            
            public double[] getGenes() { return genes.clone(); }
            public double getFitness() { return fitness; }
            
            public String toString() {
                return String.format("Individual[fitness=%.3f, genes=%s]", 
                                   fitness, Arrays.toString(Arrays.copyOf(genes, Math.min(5, genes.length))));
            }
        }
        
        public static class Population {
            private List<Individual> individuals;
            private final int populationSize;
            private final int geneLength;
            
            public Population(int populationSize, int geneLength) {
                this.populationSize = populationSize;
                this.geneLength = geneLength;
                this.individuals = new ArrayList<>();
                
                for (int i = 0; i < populationSize; i++) {
                    individuals.add(new Individual(geneLength));
                }
            }
            
            public void evaluateFitness(java.util.function.Function<double[], Double> fitnessFunction) {
                for (Individual individual : individuals) {
                    individual.calculateFitness(fitnessFunction);
                }
            }
            
            public Individual tournamentSelection(int tournamentSize) {
                Random random = new Random();
                Individual best = null;
                
                for (int i = 0; i < tournamentSize; i++) {
                    Individual candidate = individuals.get(random.nextInt(populationSize));
                    if (best == null || candidate.getFitness() > best.getFitness()) {
                        best = candidate;
                    }
                }
                
                return best;
            }
            
            public void evolve(double crossoverRate, double mutationRate, double mutationStrength, int tournamentSize) {
                List<Individual> newPopulation = new ArrayList<>();
                
                // Keep best individual (elitism)
                Individual best = getBest();
                newPopulation.add(new Individual(best.getGenes()));
                
                while (newPopulation.size() < populationSize) {
                    Individual parent1 = tournamentSelection(tournamentSize);
                    Individual parent2 = tournamentSelection(tournamentSize);
                    
                    Individual child = parent1.crossover(parent2, crossoverRate);
                    child.mutate(mutationRate, mutationStrength);
                    
                    newPopulation.add(child);
                }
                
                individuals = newPopulation;
            }
            
            public Individual getBest() {
                return individuals.stream()
                    .max(Comparator.comparingDouble(Individual::getFitness))
                    .orElse(null);
            }
            
            public double getAverageFitness() {
                return individuals.stream()
                    .mapToDouble(Individual::getFitness)
                    .average()
                    .orElse(0.0);
            }
            
            public List<Individual> getIndividuals() { return new ArrayList<>(individuals); }
        }
    }
    
    /**
     * Expert System with Rule-Based Reasoning
     */
    public static class ExpertSystem {
        
        public static class Fact {
            private final String name;
            private final Object value;
            private final double confidence;
            
            public Fact(String name, Object value, double confidence) {
                this.name = name;
                this.value = value;
                this.confidence = confidence;
            }
            
            public String getName() { return name; }
            public Object getValue() { return value; }
            public double getConfidence() { return confidence; }
            
            public String toString() {
                return String.format("Fact[%s=%s, confidence=%.2f]", name, value, confidence);
            }
        }
        
        public static class Rule {
            private final String name;
            private final List<Condition> conditions;
            private final Action action;
            private final double confidence;
            
            public Rule(String name, List<Condition> conditions, Action action, double confidence) {
                this.name = name;
                this.conditions = conditions;
                this.action = action;
                this.confidence = confidence;
            }
            
            public boolean canFire(Map<String, Fact> facts) {
                return conditions.stream().allMatch(condition -> condition.evaluate(facts));
            }
            
            public void fire(Map<String, Fact> facts) {
                if (canFire(facts)) {
                    double combinedConfidence = calculateCombinedConfidence(facts);
                    action.execute(facts, combinedConfidence * confidence);
                }
            }
            
            private double calculateCombinedConfidence(Map<String, Fact> facts) {
                double product = 1.0;
                for (Condition condition : conditions) {
                    Fact fact = facts.get(condition.getFactName());
                    if (fact != null) {
                        product *= fact.getConfidence();
                    }
                }
                return product;
            }
            
            public String getName() { return name; }
            public double getConfidence() { return confidence; }
            
            public String toString() {
                return String.format("Rule[%s: %s -> %s]", name, conditions, action);
            }
        }
        
        public interface Condition {
            boolean evaluate(Map<String, Fact> facts);
            String getFactName();
        }
        
        public static class EqualsCondition implements Condition {
            private final String factName;
            private final Object expectedValue;
            
            public EqualsCondition(String factName, Object expectedValue) {
                this.factName = factName;
                this.expectedValue = expectedValue;
            }
            
            @Override
            public boolean evaluate(Map<String, Fact> facts) {
                Fact fact = facts.get(factName);
                return fact != null && Objects.equals(fact.getValue(), expectedValue);
            }
            
            @Override
            public String getFactName() { return factName; }
            
            public String toString() {
                return String.format("%s == %s", factName, expectedValue);
            }
        }
        
        public static class GreaterThanCondition implements Condition {
            private final String factName;
            private final double threshold;
            
            public GreaterThanCondition(String factName, double threshold) {
                this.factName = factName;
                this.threshold = threshold;
            }
            
            @Override
            public boolean evaluate(Map<String, Fact> facts) {
                Fact fact = facts.get(factName);
                if (fact == null || !(fact.getValue() instanceof Number)) return false;
                return ((Number) fact.getValue()).doubleValue() > threshold;
            }
            
            @Override
            public String getFactName() { return factName; }
            
            public String toString() {
                return String.format("%s > %.2f", factName, threshold);
            }
        }
        
        public interface Action {
            void execute(Map<String, Fact> facts, double confidence);
        }
        
        public static class AddFactAction implements Action {
            private final String factName;
            private final Object value;
            
            public AddFactAction(String factName, Object value) {
                this.factName = factName;
                this.value = value;
            }
            
            @Override
            public void execute(Map<String, Fact> facts, double confidence) {
                facts.put(factName, new Fact(factName, value, confidence));
            }
            
            public String toString() {
                return String.format("ADD %s = %s", factName, value);
            }
        }
        
        public static class InferenceEngine {
            private final List<Rule> rules;
            private final Map<String, Fact> facts;
            
            public InferenceEngine() {
                this.rules = new ArrayList<>();
                this.facts = new ConcurrentHashMap<>();
            }
            
            public void addRule(Rule rule) {
                rules.add(rule);
            }
            
            public void addFact(Fact fact) {
                facts.put(fact.getName(), fact);
            }
            
            public void infer() {
                boolean changed = true;
                int iterations = 0;
                final int maxIterations = 100;
                
                while (changed && iterations < maxIterations) {
                    changed = false;
                    int initialFactCount = facts.size();
                    
                    for (Rule rule : rules) {
                        if (rule.canFire(facts)) {
                            rule.fire(facts);
                        }
                    }
                    
                    if (facts.size() > initialFactCount) {
                        changed = true;
                    }
                    
                    iterations++;
                }
                
                System.out.printf("Inference completed in %d iterations\n", iterations);
            }
            
            public Map<String, Fact> getFacts() {
                return new HashMap<>(facts);
            }
            
            public void printFacts() {
                System.out.println("Current facts:");
                facts.values().forEach(fact -> System.out.println("  " + fact));
            }
        }
    }
    
    /**
     * Monte Carlo Tree Search (MCTS)
     */
    public static class MonteCarloTreeSearch {
        
        public static class MCTSNode {
            private final GameState state;
            private MCTSNode parent;
            private final List<MCTSNode> children;
            private int visits;
            private double totalScore;
            private final Random random;
            
            public MCTSNode(GameState state, MCTSNode parent) {
                this.state = state;
                this.parent = parent;
                this.children = new ArrayList<>();
                this.visits = 0;
                this.totalScore = 0;
                this.random = new Random();
            }
            
            public double getUCB1(double explorationConstant) {
                if (visits == 0) return Double.MAX_VALUE;
                
                double exploitation = totalScore / visits;
                double exploration = explorationConstant * Math.sqrt(Math.log(parent.visits) / visits);
                
                return exploitation + exploration;
            }
            
            public MCTSNode selectChild(double explorationConstant) {
                return children.stream()
                    .max(Comparator.comparingDouble(child -> child.getUCB1(explorationConstant)))
                    .orElse(null);
            }
            
            public void expand() {
                List<GameState> possibleMoves = state.getSuccessors();
                for (GameState move : possibleMoves) {
                    children.add(new MCTSNode(move, this));
                }
            }
            
            public double simulate() {
                GameState currentState = state;
                Random simRandom = new Random();
                
                while (!currentState.isTerminal()) {
                    List<GameState> moves = currentState.getSuccessors();
                    if (moves.isEmpty()) break;
                    currentState = moves.get(simRandom.nextInt(moves.size()));
                }
                
                return currentState.evaluate();
            }
            
            public void backpropagate(double score) {
                visits++;
                totalScore += score;
                
                if (parent != null) {
                    parent.backpropagate(score);
                }
            }
            
            public MCTSNode getBestChild() {
                return children.stream()
                    .max(Comparator.comparingInt(child -> child.visits))
                    .orElse(null);
            }
            
            public boolean isLeaf() { return children.isEmpty(); }
            public GameState getState() { return state; }
            public int getVisits() { return visits; }
            public double getAverageScore() { return visits > 0 ? totalScore / visits : 0; }
            
            public String toString() {
                return String.format("MCTSNode[visits=%d, avgScore=%.3f, children=%d]", 
                                   visits, getAverageScore(), children.size());
            }
        }
        
        public static class MCTS {
            private final double explorationConstant;
            private final int maxIterations;
            
            public MCTS(double explorationConstant, int maxIterations) {
                this.explorationConstant = explorationConstant;
                this.maxIterations = maxIterations;
            }
            
            public GameState findBestMove(GameState rootState) {
                MCTSNode root = new MCTSNode(rootState, null);
                
                for (int i = 0; i < maxIterations; i++) {
                    MCTSNode selectedNode = select(root);
                    
                    if (!selectedNode.getState().isTerminal() && selectedNode.getVisits() > 0) {
                        selectedNode.expand();
                        if (!selectedNode.children.isEmpty()) {
                            selectedNode = selectedNode.children.get(0);
                        }
                    }
                    
                    double score = selectedNode.simulate();
                    selectedNode.backpropagate(score);
                }
                
                MCTSNode bestChild = root.getBestChild();
                System.out.printf("MCTS completed %d iterations. Best move has %d visits, avg score %.3f\n",
                                 maxIterations, bestChild.getVisits(), bestChild.getAverageScore());
                
                return bestChild != null ? bestChild.getState() : null;
            }
            
            private MCTSNode select(MCTSNode root) {
                MCTSNode current = root;
                
                while (!current.isLeaf() && !current.getState().isTerminal()) {
                    current = current.selectChild(explorationConstant);
                }
                
                return current;
            }
        }
    }
    
    /**
     * Constraint Satisfaction Problem (CSP) Solver
     */
    public static class CSPSolver {
        
        public static class Variable {
            private final String name;
            private final Set<Object> domain;
            private Object value;
            
            public Variable(String name, Set<Object> domain) {
                this.name = name;
                this.domain = new HashSet<>(domain);
                this.value = null;
            }
            
            public String getName() { return name; }
            public Set<Object> getDomain() { return new HashSet<>(domain); }
            public Object getValue() { return value; }
            public void setValue(Object value) { this.value = value; }
            public boolean isAssigned() { return value != null; }
            
            public void removeDomainValue(Object value) {
                domain.remove(value);
            }
            
            public void restoreDomainValue(Object value) {
                domain.add(value);
            }
            
            public String toString() {
                return String.format("Variable[%s=%s, domain=%s]", name, value, domain);
            }
        }
        
        public interface Constraint {
            boolean isSatisfied(Map<String, Variable> variables);
            Set<String> getVariables();
        }
        
        public static class AllDifferentConstraint implements Constraint {
            private final Set<String> variables;
            
            public AllDifferentConstraint(Set<String> variables) {
                this.variables = new HashSet<>(variables);
            }
            
            @Override
            public boolean isSatisfied(Map<String, Variable> vars) {
                Set<Object> assignedValues = new HashSet<>();
                
                for (String varName : variables) {
                    Variable var = vars.get(varName);
                    if (var.isAssigned()) {
                        if (assignedValues.contains(var.getValue())) {
                            return false;
                        }
                        assignedValues.add(var.getValue());
                    }
                }
                
                return true;
            }
            
            @Override
            public Set<String> getVariables() { return new HashSet<>(variables); }
            
            public String toString() {
                return String.format("AllDifferent(%s)", variables);
            }
        }
        
        public static class CSP {
            private final Map<String, Variable> variables;
            private final List<Constraint> constraints;
            
            public CSP() {
                this.variables = new HashMap<>();
                this.constraints = new ArrayList<>();
            }
            
            public void addVariable(Variable variable) {
                variables.put(variable.getName(), variable);
            }
            
            public void addConstraint(Constraint constraint) {
                constraints.add(constraint);
            }
            
            public boolean solve() {
                return backtrackSearch();
            }
            
            private boolean backtrackSearch() {
                Variable unassigned = selectUnassignedVariable();
                if (unassigned == null) {
                    return true; // All variables assigned
                }
                
                for (Object value : orderDomainValues(unassigned)) {
                    unassigned.setValue(value);
                    
                    if (isConsistent()) {
                        Map<String, Set<Object>> removedValues = forwardCheck(unassigned);
                        
                        if (removedValues != null && backtrackSearch()) {
                            return true;
                        }
                        
                        // Restore domain values
                        if (removedValues != null) {
                            restoreDomains(removedValues);
                        }
                    }
                    
                    unassigned.setValue(null);
                }
                
                return false;
            }
            
            private Variable selectUnassignedVariable() {
                // Most Constrained Variable (MRV) heuristic
                return variables.values().stream()
                    .filter(var -> !var.isAssigned())
                    .min(Comparator.comparingInt(var -> var.getDomain().size()))
                    .orElse(null);
            }
            
            private List<Object> orderDomainValues(Variable variable) {
                // Least Constraining Value heuristic
                return new ArrayList<>(variable.getDomain());
            }
            
            private boolean isConsistent() {
                return constraints.stream().allMatch(constraint -> constraint.isSatisfied(variables));
            }
            
            private Map<String, Set<Object>> forwardCheck(Variable assignedVar) {
                Map<String, Set<Object>> removedValues = new HashMap<>();
                
                for (Constraint constraint : constraints) {
                    if (constraint.getVariables().contains(assignedVar.getName())) {
                        for (String varName : constraint.getVariables()) {
                            Variable var = variables.get(varName);
                            if (!var.isAssigned()) {
                                Set<Object> toRemove = new HashSet<>();
                                
                                for (Object value : var.getDomain()) {
                                    var.setValue(value);
                                    if (!constraint.isSatisfied(variables)) {
                                        toRemove.add(value);
                                    }
                                    var.setValue(null);
                                }
                                
                                if (toRemove.size() == var.getDomain().size()) {
                                    return null; // Domain wipeout
                                }
                                
                                for (Object value : toRemove) {
                                    var.removeDomainValue(value);
                                }
                                
                                if (!toRemove.isEmpty()) {
                                    removedValues.put(varName, toRemove);
                                }
                            }
                        }
                    }
                }
                
                return removedValues;
            }
            
            private void restoreDomains(Map<String, Set<Object>> removedValues) {
                for (Map.Entry<String, Set<Object>> entry : removedValues.entrySet()) {
                    Variable var = variables.get(entry.getKey());
                    for (Object value : entry.getValue()) {
                        var.restoreDomainValue(value);
                    }
                }
            }
            
            public Map<String, Variable> getVariables() { return new HashMap<>(variables); }
            
            public void printSolution() {
                System.out.println("CSP Solution:");
                variables.values().forEach(var -> System.out.println("  " + var));
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Artificial Intelligence Algorithms Demo:");
        System.out.println("======================================");
        
        // A* Search demonstration
        System.out.println("1. A* Pathfinding Algorithm:");
        int[][] grid = {
            {0, 0, 0, 1, 0},
            {0, 1, 0, 1, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0}
        };
        
        AStarSearch.Grid pathGrid = new AStarSearch.Grid(grid);
        List<AStarSearch.Node> path = AStarSearch.findPath(pathGrid, 0, 0, 4, 4);
        
        System.out.printf("Path found with %d steps:\n", path.size());
        for (int i = 0; i < path.size(); i++) {
            System.out.printf("  Step %d: %s\n", i + 1, path.get(i));
        }
        
        // Minimax with Alpha-Beta Pruning
        System.out.println("\n2. Minimax Algorithm (Tic-Tac-Toe):");
        int[][] board = {
            {1, 0, -1},
            {0, 1, 0},
            {0, 0, 0}
        };
        
        MinimaxAlgorithm.GameState gameState = new MinimaxAlgorithm.GameState(board, -1);
        MinimaxAlgorithm.MinimaxResult result = MinimaxAlgorithm.minimax(gameState, 6, false);
        
        System.out.printf("Best move score: %d (explored %d nodes)\n", 
                         result.score, result.nodesExplored);
        
        // Genetic Algorithm
        System.out.println("\n3. Genetic Algorithm (Function Optimization):");
        
        // Optimize function: f(x) = -(x^2) + 10*x (maximum at x=5)
        java.util.function.Function<double[], Double> fitnessFunction = genes -> {
            double x = genes[0];
            return -(x * x) + 10 * x;
        };
        
        GeneticAlgorithm.Population population = new GeneticAlgorithm.Population(50, 1);
        
        for (int generation = 0; generation < 100; generation++) {
            population.evaluateFitness(fitnessFunction);
            
            if (generation % 20 == 0) {
                GeneticAlgorithm.Individual best = population.getBest();
                System.out.printf("Generation %d: Best fitness=%.3f, x=%.3f\n", 
                                 generation, best.getFitness(), best.getGenes()[0]);
            }
            
            population.evolve(0.8, 0.1, 0.5, 3);
        }
        
        // Expert System
        System.out.println("\n4. Expert System (Medical Diagnosis):");
        ExpertSystem.InferenceEngine engine = new ExpertSystem.InferenceEngine();
        
        // Add initial facts
        engine.addFact(new ExpertSystem.Fact("fever", true, 0.9));
        engine.addFact(new ExpertSystem.Fact("temperature", 38.5, 0.95));
        engine.addFact(new ExpertSystem.Fact("cough", true, 0.8));
        
        // Add rules
        List<ExpertSystem.Condition> fluConditions = Arrays.asList(
            new ExpertSystem.EqualsCondition("fever", true),
            new ExpertSystem.GreaterThanCondition("temperature", 38.0),
            new ExpertSystem.EqualsCondition("cough", true)
        );
        
        ExpertSystem.Rule fluRule = new ExpertSystem.Rule(
            "Flu Diagnosis",
            fluConditions,
            new ExpertSystem.AddFactAction("diagnosis", "flu"),
            0.85
        );
        
        engine.addRule(fluRule);
        
        // Run inference
        engine.infer();
        engine.printFacts();
        
        // Monte Carlo Tree Search
        System.out.println("\n5. Monte Carlo Tree Search:");
        int[][] mctsBoard = {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, 0}
        };
        
        MinimaxAlgorithm.GameState mctsState = new MinimaxAlgorithm.GameState(mctsBoard, -1);
        MonteCarloTreeSearch.MCTS mcts = new MonteCarloTreeSearch.MCTS(1.4, 1000);
        
        MinimaxAlgorithm.GameState bestMove = mcts.findBestMove(mctsState);
        System.out.println("MCTS found best move");
        
        // Constraint Satisfaction Problem
        System.out.println("\n6. Constraint Satisfaction Problem (N-Queens):");
        CSPSolver.CSP nQueens = new CSPSolver.CSP();
        
        // 4-Queens problem
        Set<Object> positions = IntStream.range(0, 4).boxed().collect(Collectors.toSet());
        
        for (int i = 0; i < 4; i++) {
            nQueens.addVariable(new CSPSolver.Variable("Q" + i, positions));
        }
        
        // All queens must be in different columns
        Set<String> queenVars = IntStream.range(0, 4)
            .mapToObj(i -> "Q" + i)
            .collect(Collectors.toSet());
        
        nQueens.addConstraint(new CSPSolver.AllDifferentConstraint(queenVars));
        
        boolean solved = nQueens.solve();
        System.out.println("4-Queens problem solved: " + solved);
        if (solved) {
            nQueens.printSolution();
        }
        
        System.out.println("\nAI algorithms demonstration completed!");
    }
}
