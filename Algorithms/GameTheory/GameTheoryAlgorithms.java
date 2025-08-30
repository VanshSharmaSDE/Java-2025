package Algorithms.GameTheory;

import java.util.*;

/**
 * Comprehensive Game Theory Algorithms
 * Strategic games, Nash equilibrium, auctions, mechanism design, and AI game playing
 */
public class GameTheoryAlgorithms {
    
    /**
     * Strategic Form Games and Nash Equilibrium
     */
    public static class StrategicGames {
        
        /**
         * Game representation with payoff matrices
         */
        public static class BimatrixGame {
            private double[][][] payoffs; // payoffs[player][rowStrategy][colStrategy]
            private int numPlayers = 2;
            private int[] numStrategies;
            
            public BimatrixGame(double[][] player1Payoffs, double[][] player2Payoffs) {
                numStrategies = new int[]{player1Payoffs.length, player1Payoffs[0].length};
                payoffs = new double[2][][];
                payoffs[0] = deepCopy(player1Payoffs);
                payoffs[1] = deepCopy(player2Payoffs);
            }
            
            private double[][] deepCopy(double[][] original) {
                double[][] copy = new double[original.length][];
                for (int i = 0; i < original.length; i++) {
                    copy[i] = Arrays.copyOf(original[i], original[i].length);
                }
                return copy;
            }
            
            public double getPayoff(int player, int strategy1, int strategy2) {
                return payoffs[player][strategy1][strategy2];
            }
            
            public int getNumStrategies(int player) {
                return numStrategies[player];
            }
            
            /**
             * Find pure strategy Nash equilibria
             */
            public List<int[]> findPureNashEquilibria() {
                List<int[]> equilibria = new ArrayList<>();
                
                for (int i = 0; i < numStrategies[0]; i++) {
                    for (int j = 0; j < numStrategies[1]; j++) {
                        if (isPureNashEquilibrium(i, j)) {
                            equilibria.add(new int[]{i, j});
                        }
                    }
                }
                
                return equilibria;
            }
            
            private boolean isPureNashEquilibrium(int strategy1, int strategy2) {
                // Check if player 1 has incentive to deviate
                double currentPayoff1 = getPayoff(0, strategy1, strategy2);
                for (int i = 0; i < numStrategies[0]; i++) {
                    if (getPayoff(0, i, strategy2) > currentPayoff1) {
                        return false;
                    }
                }
                
                // Check if player 2 has incentive to deviate
                double currentPayoff2 = getPayoff(1, strategy1, strategy2);
                for (int j = 0; j < numStrategies[1]; j++) {
                    if (getPayoff(1, strategy1, j) > currentPayoff2) {
                        return false;
                    }
                }
                
                return true;
            }
            
            /**
             * Find dominated strategies
             */
            public List<Integer> findDominatedStrategies(int player) {
                List<Integer> dominated = new ArrayList<>();
                int numStrat = numStrategies[player];
                
                for (int i = 0; i < numStrat; i++) {
                    for (int j = 0; j < numStrat; j++) {
                        if (i != j && isDominated(player, i, j)) {
                            dominated.add(i);
                            break;
                        }
                    }
                }
                
                return dominated;
            }
            
            private boolean isDominated(int player, int strategy1, int strategy2) {
                // Check if strategy1 is dominated by strategy2
                boolean strictlyDominated = false;
                
                if (player == 0) {
                    for (int j = 0; j < numStrategies[1]; j++) {
                        double payoff1 = getPayoff(0, strategy1, j);
                        double payoff2 = getPayoff(0, strategy2, j);
                        
                        if (payoff1 > payoff2) {
                            return false;
                        }
                        if (payoff1 < payoff2) {
                            strictlyDominated = true;
                        }
                    }
                } else {
                    for (int i = 0; i < numStrategies[0]; i++) {
                        double payoff1 = getPayoff(1, i, strategy1);
                        double payoff2 = getPayoff(1, i, strategy2);
                        
                        if (payoff1 > payoff2) {
                            return false;
                        }
                        if (payoff1 < payoff2) {
                            strictlyDominated = true;
                        }
                    }
                }
                
                return strictlyDominated;
            }
            
            /**
             * Calculate best response for a player given opponent's mixed strategy
             */
            public double[] bestResponse(int player, double[] opponentMixedStrategy) {
                int numStrat = numStrategies[player];
                double[] expectedPayoffs = new double[numStrat];
                
                // Calculate expected payoff for each pure strategy
                for (int i = 0; i < numStrat; i++) {
                    expectedPayoffs[i] = 0;
                    
                    if (player == 0) {
                        for (int j = 0; j < numStrategies[1]; j++) {
                            expectedPayoffs[i] += getPayoff(0, i, j) * opponentMixedStrategy[j];
                        }
                    } else {
                        for (int j = 0; j < numStrategies[0]; j++) {
                            expectedPayoffs[i] += getPayoff(1, j, i) * opponentMixedStrategy[j];
                        }
                    }
                }
                
                // Find best response (pure strategy)
                int bestStrategy = 0;
                for (int i = 1; i < numStrat; i++) {
                    if (expectedPayoffs[i] > expectedPayoffs[bestStrategy]) {
                        bestStrategy = i;
                    }
                }
                
                double[] bestResponseStrategy = new double[numStrat];
                bestResponseStrategy[bestStrategy] = 1.0;
                return bestResponseStrategy;
            }
        }
        
        /**
         * Mixed Strategy Nash Equilibrium using Linear Programming approach
         */
        public static class MixedStrategyEquilibrium {
            
            /**
             * Find 2x2 mixed strategy Nash equilibrium analytically
             */
            public static double[][] find2x2MixedNash(BimatrixGame game) {
                if (game.getNumStrategies(0) != 2 || game.getNumStrategies(1) != 2) {
                    throw new IllegalArgumentException("Method only works for 2x2 games");
                }
                
                // For player 1 to be indifferent between strategies:
                // p * payoff(0,0,0) + (1-p) * payoff(0,1,0) = p * payoff(0,0,1) + (1-p) * payoff(0,1,1)
                double a11 = game.getPayoff(1, 0, 0);
                double a12 = game.getPayoff(1, 0, 1);
                double a21 = game.getPayoff(1, 1, 0);
                double a22 = game.getPayoff(1, 1, 1);
                
                double denominator = (a11 - a21) - (a12 - a22);
                if (Math.abs(denominator) < 1e-10) {
                    return null; // No mixed strategy equilibrium
                }
                
                double q = (a22 - a21) / denominator; // Player 2's mixing probability
                
                // For player 2 to be indifferent:
                double b11 = game.getPayoff(0, 0, 0);
                double b12 = game.getPayoff(0, 0, 1);
                double b21 = game.getPayoff(0, 1, 0);
                double b22 = game.getPayoff(0, 1, 1);
                
                denominator = (b11 - b12) - (b21 - b22);
                if (Math.abs(denominator) < 1e-10) {
                    return null;
                }
                
                double p = (b22 - b12) / denominator; // Player 1's mixing probability
                
                // Check if probabilities are valid
                if (p < 0 || p > 1 || q < 0 || q > 1) {
                    return null;
                }
                
                return new double[][]{{p, 1-p}, {q, 1-q}};
            }
        }
    }
    
    /**
     * Extensive Form Games and Backward Induction
     */
    public static class ExtensiveFormGames {
        
        public static class GameNode {
            public int player; // -1 for terminal nodes
            public List<GameNode> children;
            public double[] payoffs; // For terminal nodes
            public String action; // Action leading to this node
            public boolean isChanceNode;
            public double[] probabilities; // For chance nodes
            
            public GameNode(int player) {
                this.player = player;
                this.children = new ArrayList<>();
                this.isChanceNode = false;
            }
            
            public GameNode(double[] payoffs) {
                this.player = -1; // Terminal node
                this.payoffs = payoffs.clone();
                this.children = new ArrayList<>();
            }
            
            public boolean isTerminal() {
                return player == -1;
            }
        }
        
        public static class GameTree {
            private GameNode root;
            private int numPlayers;
            
            public GameTree(GameNode root, int numPlayers) {
                this.root = root;
                this.numPlayers = numPlayers;
            }
            
            /**
             * Backward induction to find subgame perfect equilibrium
             */
            public double[] backwardInduction() {
                return backwardInduction(root);
            }
            
            private double[] backwardInduction(GameNode node) {
                if (node.isTerminal()) {
                    return node.payoffs.clone();
                }
                
                if (node.isChanceNode) {
                    double[] expectedPayoffs = new double[numPlayers];
                    
                    for (int i = 0; i < node.children.size(); i++) {
                        double[] childPayoffs = backwardInduction(node.children.get(i));
                        for (int j = 0; j < numPlayers; j++) {
                            expectedPayoffs[j] += node.probabilities[i] * childPayoffs[j];
                        }
                    }
                    
                    return expectedPayoffs;
                } else {
                    // Player's decision node
                    double[] bestPayoffs = null;
                    double bestValue = Double.NEGATIVE_INFINITY;
                    
                    for (GameNode child : node.children) {
                        double[] childPayoffs = backwardInduction(child);
                        
                        if (childPayoffs[node.player] > bestValue) {
                            bestValue = childPayoffs[node.player];
                            bestPayoffs = childPayoffs;
                        }
                    }
                    
                    return bestPayoffs;
                }
            }
            
            /**
             * Find optimal strategy for each player
             */
            public Map<Integer, String> findOptimalStrategies() {
                Map<Integer, String> strategies = new HashMap<>();
                findOptimalStrategies(root, strategies);
                return strategies;
            }
            
            private void findOptimalStrategies(GameNode node, Map<Integer, String> strategies) {
                if (node.isTerminal() || node.isChanceNode) {
                    return;
                }
                
                double[] bestPayoffs = null;
                double bestValue = Double.NEGATIVE_INFINITY;
                GameNode bestChild = null;
                
                for (GameNode child : node.children) {
                    double[] childPayoffs = backwardInduction(child);
                    
                    if (childPayoffs[node.player] > bestValue) {
                        bestValue = childPayoffs[node.player];
                        bestPayoffs = childPayoffs;
                        bestChild = child;
                    }
                }
                
                if (bestChild != null) {
                    strategies.put(node.player, bestChild.action);
                    findOptimalStrategies(bestChild, strategies);
                }
            }
        }
    }
    
    /**
     * Auction Theory and Mechanism Design
     */
    public static class AuctionTheory {
        
        public static class Bidder {
            public int id;
            public double valuation;
            public double bid;
            
            public Bidder(int id, double valuation) {
                this.id = id;
                this.valuation = valuation;
            }
            
            public void setBid(double bid) {
                this.bid = bid;
            }
        }
        
        /**
         * First-Price Sealed-Bid Auction
         */
        public static class FirstPriceAuction {
            private List<Bidder> bidders;
            
            public FirstPriceAuction(List<Bidder> bidders) {
                this.bidders = new ArrayList<>(bidders);
            }
            
            public AuctionResult runAuction() {
                Bidder winner = null;
                double highestBid = 0;
                
                for (Bidder bidder : bidders) {
                    if (bidder.bid > highestBid) {
                        highestBid = bidder.bid;
                        winner = bidder;
                    }
                }
                
                return new AuctionResult(winner, highestBid, highestBid);
            }
            
            /**
             * Calculate optimal bid for risk-neutral bidder in symmetric equilibrium
             */
            public double optimalBid(double valuation, int numBidders) {
                // In symmetric equilibrium with uniform distribution [0,1]
                return valuation * (numBidders - 1.0) / numBidders;
            }
        }
        
        /**
         * Second-Price (Vickrey) Auction
         */
        public static class VickreyAuction {
            private List<Bidder> bidders;
            
            public VickreyAuction(List<Bidder> bidders) {
                this.bidders = new ArrayList<>(bidders);
            }
            
            public AuctionResult runAuction() {
                if (bidders.size() < 2) {
                    return new AuctionResult(null, 0, 0);
                }
                
                // Sort by bid (descending)
                List<Bidder> sortedBidders = new ArrayList<>(bidders);
                sortedBidders.sort((a, b) -> Double.compare(b.bid, a.bid));
                
                Bidder winner = sortedBidders.get(0);
                double winningBid = sortedBidders.get(0).bid;
                double paymentPrice = sortedBidders.get(1).bid;
                
                return new AuctionResult(winner, winningBid, paymentPrice);
            }
            
            /**
             * In Vickrey auction, truthful bidding is optimal
             */
            public double optimalBid(double valuation) {
                return valuation; // Truth-telling is dominant strategy
            }
        }
        
        /**
         * Dutch (Descending Price) Auction
         */
        public static class DutchAuction {
            private List<Bidder> bidders;
            private double startingPrice;
            private double decrementRate;
            
            public DutchAuction(List<Bidder> bidders, double startingPrice, double decrementRate) {
                this.bidders = new ArrayList<>(bidders);
                this.startingPrice = startingPrice;
                this.decrementRate = decrementRate;
            }
            
            public AuctionResult runAuction() {
                double currentPrice = startingPrice;
                
                while (currentPrice > 0) {
                    for (Bidder bidder : bidders) {
                        if (currentPrice <= bidder.valuation && currentPrice <= bidder.bid) {
                            return new AuctionResult(bidder, currentPrice, currentPrice);
                        }
                    }
                    currentPrice -= decrementRate;
                }
                
                return new AuctionResult(null, 0, 0);
            }
        }
        
        /**
         * All-Pay Auction
         */
        public static class AllPayAuction {
            private List<Bidder> bidders;
            
            public AllPayAuction(List<Bidder> bidders) {
                this.bidders = new ArrayList<>(bidders);
            }
            
            public AuctionResult runAuction() {
                Bidder winner = null;
                double highestBid = 0;
                double totalRevenue = 0;
                
                for (Bidder bidder : bidders) {
                    totalRevenue += bidder.bid;
                    if (bidder.bid > highestBid) {
                        highestBid = bidder.bid;
                        winner = bidder;
                    }
                }
                
                AuctionResult result = new AuctionResult(winner, highestBid, totalRevenue);
                result.allPayRevenue = totalRevenue;
                return result;
            }
            
            /**
             * Optimal bid in all-pay auction with uniform valuations
             */
            public double optimalBid(double valuation, int numBidders) {
                // Symmetric equilibrium bid function
                return valuation * (numBidders - 1.0) / numBidders;
            }
        }
        
        public static class AuctionResult {
            public Bidder winner;
            public double winningBid;
            public double paymentPrice;
            public double allPayRevenue;
            
            public AuctionResult(Bidder winner, double winningBid, double paymentPrice) {
                this.winner = winner;
                this.winningBid = winningBid;
                this.paymentPrice = paymentPrice;
            }
            
            @Override
            public String toString() {
                if (winner == null) {
                    return "No winner";
                }
                return String.format("Winner: Bidder %d, Winning bid: %.2f, Payment: %.2f", 
                                   winner.id, winningBid, paymentPrice);
            }
        }
        
        /**
         * Revenue Comparison across auction formats
         */
        public static void revenueComparison(List<Bidder> bidders) {
            // Set bids for different auction formats
            for (Bidder bidder : bidders) {
                bidder.setBid(bidder.valuation); // Truthful for second-price
            }
            
            VickreyAuction vickrey = new VickreyAuction(bidders);
            AuctionResult vickreyResult = vickrey.runAuction();
            
            // Reset bids for first-price
            for (Bidder bidder : bidders) {
                FirstPriceAuction fpa = new FirstPriceAuction(bidders);
                bidder.setBid(fpa.optimalBid(bidder.valuation, bidders.size()));
            }
            
            FirstPriceAuction firstPrice = new FirstPriceAuction(bidders);
            AuctionResult firstPriceResult = firstPrice.runAuction();
            
            System.out.println("Revenue Comparison:");
            System.out.println("Vickrey: " + vickreyResult.paymentPrice);
            System.out.println("First-Price: " + firstPriceResult.paymentPrice);
        }
    }
    
    /**
     * Evolutionary Game Theory
     */
    public static class EvolutionaryGameTheory {
        
        /**
         * Replicator Dynamics for population evolution
         */
        public static class ReplicatorDynamics {
            private double[][] payoffMatrix;
            private double[] population;
            private int numStrategies;
            
            public ReplicatorDynamics(double[][] payoffMatrix, double[] initialPopulation) {
                this.payoffMatrix = deepCopy(payoffMatrix);
                this.population = initialPopulation.clone();
                this.numStrategies = payoffMatrix.length;
            }
            
            private double[][] deepCopy(double[][] original) {
                double[][] copy = new double[original.length][];
                for (int i = 0; i < original.length; i++) {
                    copy[i] = Arrays.copyOf(original[i], original[i].length);
                }
                return copy;
            }
            
            /**
             * Simulate replicator dynamics for given number of steps
             */
            public double[][] simulate(int steps, double stepSize) {
                double[][] trajectory = new double[steps + 1][numStrategies];
                trajectory[0] = population.clone();
                
                for (int t = 0; t < steps; t++) {
                    double[] newPopulation = new double[numStrategies];
                    double averageFitness = calculateAverageFitness();
                    
                    for (int i = 0; i < numStrategies; i++) {
                        double fitness = calculateFitness(i);
                        double growthRate = stepSize * population[i] * (fitness - averageFitness);
                        newPopulation[i] = Math.max(0, population[i] + growthRate);
                    }
                    
                    // Normalize population
                    double total = Arrays.stream(newPopulation).sum();
                    if (total > 0) {
                        for (int i = 0; i < numStrategies; i++) {
                            newPopulation[i] /= total;
                        }
                    }
                    
                    population = newPopulation;
                    trajectory[t + 1] = population.clone();
                }
                
                return trajectory;
            }
            
            private double calculateFitness(int strategy) {
                double fitness = 0;
                for (int j = 0; j < numStrategies; j++) {
                    fitness += payoffMatrix[strategy][j] * population[j];
                }
                return fitness;
            }
            
            private double calculateAverageFitness() {
                double averageFitness = 0;
                for (int i = 0; i < numStrategies; i++) {
                    averageFitness += population[i] * calculateFitness(i);
                }
                return averageFitness;
            }
            
            /**
             * Find evolutionarily stable strategies (ESS)
             */
            public List<Integer> findESS() {
                List<Integer> essStrategies = new ArrayList<>();
                
                for (int i = 0; i < numStrategies; i++) {
                    if (isESS(i)) {
                        essStrategies.add(i);
                    }
                }
                
                return essStrategies;
            }
            
            private boolean isESS(int strategy) {
                double epsilon = 0.01;
                
                for (int j = 0; j < numStrategies; j++) {
                    if (i == j) continue;
                    
                    // Test invasion by strategy j
                    double[] testPopulation = new double[numStrategies];
                    testPopulation[strategy] = 1 - epsilon;
                    testPopulation[j] = epsilon;
                    
                    double fitnessI = 0, fitnessJ = 0;
                    for (int k = 0; k < numStrategies; k++) {
                        fitnessI += payoffMatrix[strategy][k] * testPopulation[k];
                        fitnessJ += payoffMatrix[j][k] * testPopulation[k];
                    }
                    
                    if (fitnessJ >= fitnessI) {
                        return false; // Strategy j can invade
                    }
                }
                
                return true;
            }
        }
        
        /**
         * Hawks and Doves game simulation
         */
        public static class HawkDoveGame {
            private double V; // Value of resource
            private double C; // Cost of fighting
            
            public HawkDoveGame(double V, double C) {
                this.V = V;
                this.C = C;
            }
            
            public double[][] createPayoffMatrix() {
                // Hawks vs Hawks: (V-C)/2
                // Hawks vs Doves: V
                // Doves vs Hawks: 0
                // Doves vs Doves: V/2
                
                return new double[][] {
                    {(V - C) / 2, V},      // Hawk's payoffs
                    {0, V / 2}             // Dove's payoffs
                };
            }
            
            public double findESSMixingProbability() {
                if (C <= V) {
                    return 1.0; // Pure Hawk strategy
                } else {
                    return V / C; // Mixed strategy
                }
            }
        }
    }
    
    /**
     * Cooperative Game Theory
     */
    public static class CooperativeGameTheory {
        
        /**
         * Shapley Value calculation
         */
        public static class ShapleyValue {
            private int numPlayers;
            private CharacteristicFunction characteristicFunction;
            
            public ShapleyValue(int numPlayers, CharacteristicFunction cf) {
                this.numPlayers = numPlayers;
                this.characteristicFunction = cf;
            }
            
            /**
             * Calculate Shapley value for all players
             */
            public double[] calculateShapleyValues() {
                double[] shapleyValues = new double[numPlayers];
                
                for (int player = 0; player < numPlayers; player++) {
                    shapleyValues[player] = calculateShapleyValue(player);
                }
                
                return shapleyValues;
            }
            
            private double calculateShapleyValue(int player) {
                double shapleyValue = 0;
                
                // Iterate over all possible coalitions not containing the player
                for (int coalitionMask = 0; coalitionMask < (1 << numPlayers); coalitionMask++) {
                    if ((coalitionMask & (1 << player)) == 0) { // Coalition doesn't contain player
                        int coalitionSize = Integer.bitCount(coalitionMask);
                        
                        double weight = factorial(coalitionSize) * factorial(numPlayers - coalitionSize - 1) 
                                       / (double) factorial(numPlayers);
                        
                        double marginalContribution = characteristicFunction.getValue(coalitionMask | (1 << player))
                                                    - characteristicFunction.getValue(coalitionMask);
                        
                        shapleyValue += weight * marginalContribution;
                    }
                }
                
                return shapleyValue;
            }
            
            private long factorial(int n) {
                long result = 1;
                for (int i = 2; i <= n; i++) {
                    result *= i;
                }
                return result;
            }
        }
        
        /**
         * Core solution concept
         */
        public static class Core {
            private int numPlayers;
            private CharacteristicFunction characteristicFunction;
            
            public Core(int numPlayers, CharacteristicFunction cf) {
                this.numPlayers = numPlayers;
                this.characteristicFunction = cf;
            }
            
            /**
             * Check if a payoff vector is in the core
             */
            public boolean isInCore(double[] payoffs) {
                // Check individual rationality
                for (int player = 0; player < numPlayers; player++) {
                    if (payoffs[player] < characteristicFunction.getValue(1 << player)) {
                        return false;
                    }
                }
                
                // Check group rationality
                double totalPayoff = Arrays.stream(payoffs).sum();
                if (Math.abs(totalPayoff - characteristicFunction.getValue((1 << numPlayers) - 1)) > 1e-10) {
                    return false;
                }
                
                // Check coalition rationality
                for (int coalitionMask = 1; coalitionMask < (1 << numPlayers) - 1; coalitionMask++) {
                    double coalitionPayoff = 0;
                    for (int player = 0; player < numPlayers; player++) {
                        if ((coalitionMask & (1 << player)) != 0) {
                            coalitionPayoff += payoffs[player];
                        }
                    }
                    
                    if (coalitionPayoff < characteristicFunction.getValue(coalitionMask)) {
                        return false;
                    }
                }
                
                return true;
            }
        }
        
        /**
         * Interface for characteristic function
         */
        public interface CharacteristicFunction {
            double getValue(int coalitionMask);
        }
        
        /**
         * Example: Voting game
         */
        public static class VotingGame implements CharacteristicFunction {
            private int[] weights;
            private int quota;
            
            public VotingGame(int[] weights, int quota) {
                this.weights = weights.clone();
                this.quota = quota;
            }
            
            @Override
            public double getValue(int coalitionMask) {
                int totalWeight = 0;
                for (int i = 0; i < weights.length; i++) {
                    if ((coalitionMask & (1 << i)) != 0) {
                        totalWeight += weights[i];
                    }
                }
                
                return totalWeight >= quota ? 1.0 : 0.0;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Game Theory Algorithms Demo:");
        System.out.println("============================");
        
        // Strategic Games Demo
        System.out.println("1. Strategic Form Games:");
        
        // Prisoner's Dilemma
        double[][] player1Payoffs = {{3, 0}, {5, 1}};
        double[][] player2Payoffs = {{3, 5}, {0, 1}};
        
        StrategicGames.BimatrixGame prisonersDilemma = 
            new StrategicGames.BimatrixGame(player1Payoffs, player2Payoffs);
        
        List<int[]> pureNash = prisonersDilemma.findPureNashEquilibria();
        System.out.println("Prisoner's Dilemma Pure Nash Equilibria:");
        for (int[] equilibrium : pureNash) {
            System.out.println("(" + equilibrium[0] + ", " + equilibrium[1] + ")");
        }
        
        // Battle of Sexes - Mixed Strategy
        double[][] bos1 = {{2, 0}, {0, 1}};
        double[][] bos2 = {{1, 0}, {0, 2}};
        StrategicGames.BimatrixGame battleOfSexes = 
            new StrategicGames.BimatrixGame(bos1, bos2);
        
        double[][] mixedNash = StrategicGames.MixedStrategyEquilibrium.find2x2MixedNash(battleOfSexes);
        if (mixedNash != null) {
            System.out.println("Battle of Sexes Mixed Nash:");
            System.out.printf("Player 1: (%.3f, %.3f)\n", mixedNash[0][0], mixedNash[0][1]);
            System.out.printf("Player 2: (%.3f, %.3f)\n", mixedNash[1][0], mixedNash[1][1]);
        }
        
        // Auction Theory Demo
        System.out.println("\n2. Auction Theory:");
        
        List<AuctionTheory.Bidder> bidders = Arrays.asList(
            new AuctionTheory.Bidder(1, 100),
            new AuctionTheory.Bidder(2, 80),
            new AuctionTheory.Bidder(3, 120),
            new AuctionTheory.Bidder(4, 90)
        );
        
        // Vickrey Auction
        for (AuctionTheory.Bidder bidder : bidders) {
            bidder.setBid(bidder.valuation); // Truth-telling
        }
        
        AuctionTheory.VickreyAuction vickrey = new AuctionTheory.VickreyAuction(bidders);
        AuctionTheory.AuctionResult vickreyResult = vickrey.runAuction();
        System.out.println("Vickrey Auction: " + vickreyResult);
        
        // First-Price Auction
        AuctionTheory.FirstPriceAuction fpa = new AuctionTheory.FirstPriceAuction(bidders);
        for (AuctionTheory.Bidder bidder : bidders) {
            bidder.setBid(fpa.optimalBid(bidder.valuation, bidders.size()));
        }
        
        AuctionTheory.AuctionResult firstPriceResult = fpa.runAuction();
        System.out.println("First-Price Auction: " + firstPriceResult);
        
        // Evolutionary Game Theory Demo
        System.out.println("\n3. Evolutionary Game Theory:");
        
        EvolutionaryGameTheory.HawkDoveGame hawkDove = 
            new EvolutionaryGameTheory.HawkDoveGame(10, 15);
        double[][] hawkDoveMatrix = hawkDove.createPayoffMatrix();
        
        double[] initialPopulation = {0.3, 0.7}; // 30% Hawks, 70% Doves
        EvolutionaryGameTheory.ReplicatorDynamics replicator = 
            new EvolutionaryGameTheory.ReplicatorDynamics(hawkDoveMatrix, initialPopulation);
        
        double[][] trajectory = replicator.simulate(100, 0.01);
        System.out.printf("Hawk-Dove Evolution - Initial: (%.2f, %.2f), Final: (%.2f, %.2f)\n",
                         trajectory[0][0], trajectory[0][1],
                         trajectory[100][0], trajectory[100][1]);
        
        double essMixing = hawkDove.findESSMixingProbability();
        System.out.printf("ESS Hawk probability: %.3f\n", essMixing);
        
        // Cooperative Game Theory Demo
        System.out.println("\n4. Cooperative Games:");
        
        // Voting game example
        int[] votingWeights = {6, 4, 3, 2};
        int quota = 8;
        CooperativeGameTheory.VotingGame votingGame = 
            new CooperativeGameTheory.VotingGame(votingWeights, quota);
        
        CooperativeGameTheory.ShapleyValue shapley = 
            new CooperativeGameTheory.ShapleyValue(4, votingGame);
        double[] shapleyValues = shapley.calculateShapleyValues();
        
        System.out.println("Voting Game Shapley Values:");
        for (int i = 0; i < shapleyValues.length; i++) {
            System.out.printf("Player %d: %.3f\n", i + 1, shapleyValues[i]);
        }
        
        // Core check
        CooperativeGameTheory.Core core = new CooperativeGameTheory.Core(4, votingGame);
        boolean inCore = core.isInCore(shapleyValues);
        System.out.println("Shapley values in core: " + inCore);
        
        // Extensive Form Games Demo
        System.out.println("\n5. Extensive Form Games:");
        
        // Simple 2-player sequential game
        ExtensiveFormGames.GameNode root = new ExtensiveFormGames.GameNode(0); // Player 0 moves first
        ExtensiveFormGames.GameNode left = new ExtensiveFormGames.GameNode(1); // Player 1 moves
        ExtensiveFormGames.GameNode right = new ExtensiveFormGames.GameNode(1); // Player 1 moves
        
        // Terminal nodes
        ExtensiveFormGames.GameNode ll = new ExtensiveFormGames.GameNode(new double[]{2, 1});
        ExtensiveFormGames.GameNode lr = new ExtensiveFormGames.GameNode(new double[]{0, 0});
        ExtensiveFormGames.GameNode rl = new ExtensiveFormGames.GameNode(new double[]{1, 2});
        ExtensiveFormGames.GameNode rr = new ExtensiveFormGames.GameNode(new double[]{3, 1});
        
        // Build tree
        root.children.add(left);
        root.children.add(right);
        left.children.add(ll);
        left.children.add(lr);
        right.children.add(rl);
        right.children.add(rr);
        
        // Set actions
        left.action = "Left";
        right.action = "Right";
        ll.action = "Left-Left";
        lr.action = "Left-Right";
        rl.action = "Right-Left";
        rr.action = "Right-Right";
        
        ExtensiveFormGames.GameTree gameTree = new ExtensiveFormGames.GameTree(root, 2);
        double[] backwardInductionPayoffs = gameTree.backwardInduction();
        System.out.printf("Backward Induction Payoffs: (%.1f, %.1f)\n", 
                         backwardInductionPayoffs[0], backwardInductionPayoffs[1]);
        
        Map<Integer, String> optimalStrategies = gameTree.findOptimalStrategies();
        System.out.println("Optimal Strategies:");
        for (Map.Entry<Integer, String> entry : optimalStrategies.entrySet()) {
            System.out.println("Player " + entry.getKey() + ": " + entry.getValue());
        }
        
        System.out.println("\nGame Theory demonstration completed!");
    }
}
