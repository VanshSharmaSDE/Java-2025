package Algorithms.Optimization;

import java.util.*;
import java.util.function.*;

/**
 * Comprehensive Optimization Algorithms
 * Linear programming, non-linear optimization, metaheuristics, and specialized optimization techniques
 */
public class OptimizationAlgorithms {
    
    /**
     * Linear Programming using Simplex Method
     */
    public static class SimplexMethod {
        private double[][] tableau;
        private int numVariables;
        private int numConstraints;
        private boolean isMaximization;
        
        public SimplexMethod(double[] objectiveFunction, double[][] constraints, double[] rhs, boolean isMaximization) {
            this.numVariables = objectiveFunction.length;
            this.numConstraints = constraints.length;
            this.isMaximization = isMaximization;
            
            // Create tableau: [constraints | slack variables | rhs]
            // Bottom row: [objective function | 0s for slack | 0]
            int tableauCols = numVariables + numConstraints + 1; // +1 for RHS
            int tableauRows = numConstraints + 1; // +1 for objective function
            
            tableau = new double[tableauRows][tableauCols];
            
            // Fill constraints
            for (int i = 0; i < numConstraints; i++) {
                // Copy constraint coefficients
                System.arraycopy(constraints[i], 0, tableau[i], 0, numVariables);
                
                // Add slack variable (identity matrix)
                tableau[i][numVariables + i] = 1.0;
                
                // RHS
                tableau[i][tableauCols - 1] = rhs[i];
            }
            
            // Fill objective function (bottom row)
            for (int j = 0; j < numVariables; j++) {
                tableau[numConstraints][j] = isMaximization ? -objectiveFunction[j] : objectiveFunction[j];
            }
        }
        
        public OptimizationResult solve() {
            int maxIterations = 1000;
            int iterations = 0;
            
            while (iterations < maxIterations && !isOptimal()) {
                int pivotCol = findPivotColumn();
                if (pivotCol == -1) {
                    return new OptimizationResult(null, Double.NaN, OptimizationStatus.UNBOUNDED);
                }
                
                int pivotRow = findPivotRow(pivotCol);
                if (pivotRow == -1) {
                    return new OptimizationResult(null, Double.POSITIVE_INFINITY, OptimizationStatus.UNBOUNDED);
                }
                
                pivot(pivotRow, pivotCol);
                iterations++;
            }
            
            if (iterations >= maxIterations) {
                return new OptimizationResult(null, Double.NaN, OptimizationStatus.MAX_ITERATIONS);
            }
            
            return extractSolution();
        }
        
        private boolean isOptimal() {
            int objectiveRow = numConstraints;
            for (int j = 0; j < numVariables + numConstraints; j++) {
                if (tableau[objectiveRow][j] < -1e-10) {
                    return false;
                }
            }
            return true;
        }
        
        private int findPivotColumn() {
            int objectiveRow = numConstraints;
            int pivotCol = -1;
            double mostNegative = 0;
            
            for (int j = 0; j < numVariables + numConstraints; j++) {
                if (tableau[objectiveRow][j] < mostNegative) {
                    mostNegative = tableau[objectiveRow][j];
                    pivotCol = j;
                }
            }
            
            return pivotCol;
        }
        
        private int findPivotRow(int pivotCol) {
            int pivotRow = -1;
            double minRatio = Double.POSITIVE_INFINITY;
            int rhsCol = tableau[0].length - 1;
            
            for (int i = 0; i < numConstraints; i++) {
                if (tableau[i][pivotCol] > 1e-10) {
                    double ratio = tableau[i][rhsCol] / tableau[i][pivotCol];
                    if (ratio >= 0 && ratio < minRatio) {
                        minRatio = ratio;
                        pivotRow = i;
                    }
                }
            }
            
            return pivotRow;
        }
        
        private void pivot(int pivotRow, int pivotCol) {
            double pivotElement = tableau[pivotRow][pivotCol];
            
            // Normalize pivot row
            for (int j = 0; j < tableau[pivotRow].length; j++) {
                tableau[pivotRow][j] /= pivotElement;
            }
            
            // Eliminate other rows
            for (int i = 0; i < tableau.length; i++) {
                if (i != pivotRow) {
                    double multiplier = tableau[i][pivotCol];
                    for (int j = 0; j < tableau[i].length; j++) {
                        tableau[i][j] -= multiplier * tableau[pivotRow][j];
                    }
                }
            }
        }
        
        private OptimizationResult extractSolution() {
            double[] solution = new double[numVariables];
            int rhsCol = tableau[0].length - 1;
            
            // Find basic variables
            for (int j = 0; j < numVariables; j++) {
                int basicRow = -1;
                int nonZeroCount = 0;
                
                for (int i = 0; i < numConstraints; i++) {
                    if (Math.abs(tableau[i][j]) > 1e-10) {
                        if (Math.abs(tableau[i][j] - 1.0) < 1e-10) {
                            basicRow = i;
                        }
                        nonZeroCount++;
                    }
                }
                
                if (nonZeroCount == 1 && basicRow != -1) {
                    solution[j] = tableau[basicRow][rhsCol];
                }
            }
            
            double objectiveValue = tableau[numConstraints][rhsCol];
            if (isMaximization) {
                objectiveValue = -objectiveValue;
            }
            
            return new OptimizationResult(solution, objectiveValue, OptimizationStatus.OPTIMAL);
        }
    }
    
    /**
     * Genetic Algorithm for Global Optimization
     */
    public static class GeneticAlgorithm {
        private Function<double[], Double> objectiveFunction;
        private double[] lowerBounds;
        private double[] upperBounds;
        private int populationSize;
        private double mutationRate;
        private double crossoverRate;
        private int maxGenerations;
        private Random random;
        
        public GeneticAlgorithm(Function<double[], Double> objectiveFunction, 
                               double[] lowerBounds, double[] upperBounds,
                               int populationSize, double mutationRate, double crossoverRate, int maxGenerations) {
            this.objectiveFunction = objectiveFunction;
            this.lowerBounds = lowerBounds.clone();
            this.upperBounds = upperBounds.clone();
            this.populationSize = populationSize;
            this.mutationRate = mutationRate;
            this.crossoverRate = crossoverRate;
            this.maxGenerations = maxGenerations;
            this.random = new Random();
        }
        
        public OptimizationResult optimize() {
            List<Individual> population = initializePopulation();
            
            Individual bestIndividual = null;
            double bestFitness = Double.POSITIVE_INFINITY;
            
            for (int generation = 0; generation < maxGenerations; generation++) {
                // Evaluate population
                for (Individual individual : population) {
                    individual.fitness = objectiveFunction.apply(individual.genes);
                    
                    if (individual.fitness < bestFitness) {
                        bestFitness = individual.fitness;
                        bestIndividual = new Individual(individual.genes.clone());
                        bestIndividual.fitness = individual.fitness;
                    }
                }
                
                // Create next generation
                List<Individual> newPopulation = new ArrayList<>();
                
                // Elitism - keep best individual
                newPopulation.add(new Individual(bestIndividual.genes.clone()));
                
                while (newPopulation.size() < populationSize) {
                    Individual parent1 = tournamentSelection(population);
                    Individual parent2 = tournamentSelection(population);
                    
                    Individual[] offspring = crossover(parent1, parent2);
                    
                    mutate(offspring[0]);
                    mutate(offspring[1]);
                    
                    newPopulation.add(offspring[0]);
                    if (newPopulation.size() < populationSize) {
                        newPopulation.add(offspring[1]);
                    }
                }
                
                population = newPopulation;
                
                if (generation % 100 == 0) {
                    System.out.printf("Generation %d: Best fitness = %.6f\n", generation, bestFitness);
                }
            }
            
            return new OptimizationResult(bestIndividual.genes, bestFitness, OptimizationStatus.OPTIMAL);
        }
        
        private List<Individual> initializePopulation() {
            List<Individual> population = new ArrayList<>();
            
            for (int i = 0; i < populationSize; i++) {
                double[] genes = new double[lowerBounds.length];
                for (int j = 0; j < genes.length; j++) {
                    genes[j] = lowerBounds[j] + random.nextDouble() * (upperBounds[j] - lowerBounds[j]);
                }
                population.add(new Individual(genes));
            }
            
            return population;
        }
        
        private Individual tournamentSelection(List<Individual> population) {
            int tournamentSize = 3;
            Individual best = null;
            
            for (int i = 0; i < tournamentSize; i++) {
                Individual candidate = population.get(random.nextInt(population.size()));
                if (best == null || candidate.fitness < best.fitness) {
                    best = candidate;
                }
            }
            
            return best;
        }
        
        private Individual[] crossover(Individual parent1, Individual parent2) {
            double[] offspring1 = new double[parent1.genes.length];
            double[] offspring2 = new double[parent2.genes.length];
            
            if (random.nextDouble() < crossoverRate) {
                // Arithmetic crossover
                double alpha = random.nextDouble();
                for (int i = 0; i < offspring1.length; i++) {
                    offspring1[i] = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i];
                    offspring2[i] = (1 - alpha) * parent1.genes[i] + alpha * parent2.genes[i];
                }
            } else {
                System.arraycopy(parent1.genes, 0, offspring1, 0, offspring1.length);
                System.arraycopy(parent2.genes, 0, offspring2, 0, offspring2.length);
            }
            
            return new Individual[]{new Individual(offspring1), new Individual(offspring2)};
        }
        
        private void mutate(Individual individual) {
            for (int i = 0; i < individual.genes.length; i++) {
                if (random.nextDouble() < mutationRate) {
                    // Gaussian mutation
                    double mutation = random.nextGaussian() * 0.1 * (upperBounds[i] - lowerBounds[i]);
                    individual.genes[i] += mutation;
                    
                    // Ensure bounds
                    individual.genes[i] = Math.max(lowerBounds[i], 
                                                  Math.min(upperBounds[i], individual.genes[i]));
                }
            }
        }
        
        private static class Individual {
            double[] genes;
            double fitness;
            
            Individual(double[] genes) {
                this.genes = genes;
                this.fitness = Double.POSITIVE_INFINITY;
            }
        }
    }
    
    /**
     * Simulated Annealing
     */
    public static class SimulatedAnnealing {
        private Function<double[], Double> objectiveFunction;
        private double[] lowerBounds;
        private double[] upperBounds;
        private double initialTemperature;
        private double finalTemperature;
        private double coolingRate;
        private int maxIterations;
        private Random random;
        
        public SimulatedAnnealing(Function<double[], Double> objectiveFunction,
                                 double[] lowerBounds, double[] upperBounds,
                                 double initialTemperature, double finalTemperature,
                                 double coolingRate, int maxIterations) {
            this.objectiveFunction = objectiveFunction;
            this.lowerBounds = lowerBounds.clone();
            this.upperBounds = upperBounds.clone();
            this.initialTemperature = initialTemperature;
            this.finalTemperature = finalTemperature;
            this.coolingRate = coolingRate;
            this.maxIterations = maxIterations;
            this.random = new Random();
        }
        
        public OptimizationResult optimize(double[] initialSolution) {
            double[] currentSolution = initialSolution.clone();
            double currentEnergy = objectiveFunction.apply(currentSolution);
            
            double[] bestSolution = currentSolution.clone();
            double bestEnergy = currentEnergy;
            
            double temperature = initialTemperature;
            
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                // Generate neighbor solution
                double[] newSolution = generateNeighbor(currentSolution);
                double newEnergy = objectiveFunction.apply(newSolution);
                
                // Accept or reject new solution
                if (newEnergy < currentEnergy || 
                    random.nextDouble() < Math.exp(-(newEnergy - currentEnergy) / temperature)) {
                    currentSolution = newSolution;
                    currentEnergy = newEnergy;
                    
                    if (newEnergy < bestEnergy) {
                        bestSolution = newSolution.clone();
                        bestEnergy = newEnergy;
                    }
                }
                
                // Cool down
                temperature = Math.max(finalTemperature, temperature * coolingRate);
                
                if (iteration % 1000 == 0) {
                    System.out.printf("Iteration %d: Best = %.6f, Current = %.6f, Temp = %.6f\n",
                                     iteration, bestEnergy, currentEnergy, temperature);
                }
            }
            
            return new OptimizationResult(bestSolution, bestEnergy, OptimizationStatus.OPTIMAL);
        }
        
        private double[] generateNeighbor(double[] solution) {
            double[] neighbor = solution.clone();
            int dimension = random.nextInt(solution.length);
            
            // Add Gaussian noise
            double stepSize = 0.1 * (upperBounds[dimension] - lowerBounds[dimension]);
            neighbor[dimension] += random.nextGaussian() * stepSize;
            
            // Ensure bounds
            neighbor[dimension] = Math.max(lowerBounds[dimension],
                                         Math.min(upperBounds[dimension], neighbor[dimension]));
            
            return neighbor;
        }
    }
    
    /**
     * Particle Swarm Optimization
     */
    public static class ParticleSwarmOptimization {
        private Function<double[], Double> objectiveFunction;
        private double[] lowerBounds;
        private double[] upperBounds;
        private int numParticles;
        private int maxIterations;
        private double w; // Inertia weight
        private double c1; // Cognitive parameter
        private double c2; // Social parameter
        private Random random;
        
        public ParticleSwarmOptimization(Function<double[], Double> objectiveFunction,
                                       double[] lowerBounds, double[] upperBounds,
                                       int numParticles, int maxIterations) {
            this.objectiveFunction = objectiveFunction;
            this.lowerBounds = lowerBounds.clone();
            this.upperBounds = upperBounds.clone();
            this.numParticles = numParticles;
            this.maxIterations = maxIterations;
            this.w = 0.7;
            this.c1 = 1.5;
            this.c2 = 1.5;
            this.random = new Random();
        }
        
        public OptimizationResult optimize() {
            List<Particle> particles = initializeSwarm();
            double[] globalBest = null;
            double globalBestFitness = Double.POSITIVE_INFINITY;
            
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                for (Particle particle : particles) {
                    double fitness = objectiveFunction.apply(particle.position);
                    
                    // Update personal best
                    if (fitness < particle.bestFitness) {
                        particle.bestFitness = fitness;
                        particle.bestPosition = particle.position.clone();
                    }
                    
                    // Update global best
                    if (fitness < globalBestFitness) {
                        globalBestFitness = fitness;
                        globalBest = particle.position.clone();
                    }
                }
                
                // Update velocities and positions
                for (Particle particle : particles) {
                    updateParticle(particle, globalBest);
                }
                
                if (iteration % 100 == 0) {
                    System.out.printf("Iteration %d: Global best = %.6f\n", iteration, globalBestFitness);
                }
            }
            
            return new OptimizationResult(globalBest, globalBestFitness, OptimizationStatus.OPTIMAL);
        }
        
        private List<Particle> initializeSwarm() {
            List<Particle> particles = new ArrayList<>();
            
            for (int i = 0; i < numParticles; i++) {
                double[] position = new double[lowerBounds.length];
                double[] velocity = new double[lowerBounds.length];
                
                for (int j = 0; j < position.length; j++) {
                    position[j] = lowerBounds[j] + random.nextDouble() * (upperBounds[j] - lowerBounds[j]);
                    velocity[j] = (random.nextDouble() - 0.5) * (upperBounds[j] - lowerBounds[j]) * 0.1;
                }
                
                particles.add(new Particle(position, velocity));
            }
            
            return particles;
        }
        
        private void updateParticle(Particle particle, double[] globalBest) {
            for (int i = 0; i < particle.position.length; i++) {
                double r1 = random.nextDouble();
                double r2 = random.nextDouble();
                
                // Update velocity
                particle.velocity[i] = w * particle.velocity[i] +
                                     c1 * r1 * (particle.bestPosition[i] - particle.position[i]) +
                                     c2 * r2 * (globalBest[i] - particle.position[i]);
                
                // Update position
                particle.position[i] += particle.velocity[i];
                
                // Ensure bounds
                if (particle.position[i] < lowerBounds[i]) {
                    particle.position[i] = lowerBounds[i];
                    particle.velocity[i] = 0;
                }
                if (particle.position[i] > upperBounds[i]) {
                    particle.position[i] = upperBounds[i];
                    particle.velocity[i] = 0;
                }
            }
        }
        
        private static class Particle {
            double[] position;
            double[] velocity;
            double[] bestPosition;
            double bestFitness;
            
            Particle(double[] position, double[] velocity) {
                this.position = position;
                this.velocity = velocity;
                this.bestPosition = position.clone();
                this.bestFitness = Double.POSITIVE_INFINITY;
            }
        }
    }
    
    /**
     * Gradient Descent for Continuous Optimization
     */
    public static class GradientDescent {
        private Function<double[], Double> objectiveFunction;
        private Function<double[], double[]> gradientFunction;
        private double learningRate;
        private int maxIterations;
        private double tolerance;
        
        public GradientDescent(Function<double[], Double> objectiveFunction,
                              Function<double[], double[]> gradientFunction,
                              double learningRate, int maxIterations, double tolerance) {
            this.objectiveFunction = objectiveFunction;
            this.gradientFunction = gradientFunction;
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
        }
        
        public OptimizationResult optimize(double[] initialSolution) {
            double[] x = initialSolution.clone();
            
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                double[] gradient = gradientFunction.apply(x);
                double gradientNorm = 0;
                
                // Update solution
                for (int i = 0; i < x.length; i++) {
                    x[i] -= learningRate * gradient[i];
                    gradientNorm += gradient[i] * gradient[i];
                }
                
                gradientNorm = Math.sqrt(gradientNorm);
                
                if (iteration % 100 == 0) {
                    double value = objectiveFunction.apply(x);
                    System.out.printf("Iteration %d: f(x) = %.6f, ||grad|| = %.6f\n", 
                                     iteration, value, gradientNorm);
                }
                
                // Check convergence
                if (gradientNorm < tolerance) {
                    double finalValue = objectiveFunction.apply(x);
                    return new OptimizationResult(x, finalValue, OptimizationStatus.OPTIMAL);
                }
            }
            
            double finalValue = objectiveFunction.apply(x);
            return new OptimizationResult(x, finalValue, OptimizationStatus.MAX_ITERATIONS);
        }
        
        /**
         * Numerical gradient approximation
         */
        public static Function<double[], double[]> numericalGradient(Function<double[], Double> f, double h) {
            return x -> {
                double[] gradient = new double[x.length];
                
                for (int i = 0; i < x.length; i++) {
                    double[] xPlus = x.clone();
                    double[] xMinus = x.clone();
                    
                    xPlus[i] += h;
                    xMinus[i] -= h;
                    
                    gradient[i] = (f.apply(xPlus) - f.apply(xMinus)) / (2 * h);
                }
                
                return gradient;
            };
        }
    }
    
    /**
     * Differential Evolution
     */
    public static class DifferentialEvolution {
        private Function<double[], Double> objectiveFunction;
        private double[] lowerBounds;
        private double[] upperBounds;
        private int populationSize;
        private double mutationFactor;
        private double crossoverRate;
        private int maxGenerations;
        private Random random;
        
        public DifferentialEvolution(Function<double[], Double> objectiveFunction,
                                   double[] lowerBounds, double[] upperBounds,
                                   int populationSize, double mutationFactor, 
                                   double crossoverRate, int maxGenerations) {
            this.objectiveFunction = objectiveFunction;
            this.lowerBounds = lowerBounds.clone();
            this.upperBounds = upperBounds.clone();
            this.populationSize = populationSize;
            this.mutationFactor = mutationFactor;
            this.crossoverRate = crossoverRate;
            this.maxGenerations = maxGenerations;
            this.random = new Random();
        }
        
        public OptimizationResult optimize() {
            List<double[]> population = initializePopulation();
            double[] bestSolution = null;
            double bestFitness = Double.POSITIVE_INFINITY;
            
            for (int generation = 0; generation < maxGenerations; generation++) {
                List<double[]> newPopulation = new ArrayList<>();
                
                for (int i = 0; i < populationSize; i++) {
                    double[] target = population.get(i);
                    double[] trial = createTrialVector(population, i);
                    
                    double targetFitness = objectiveFunction.apply(target);
                    double trialFitness = objectiveFunction.apply(trial);
                    
                    if (trialFitness <= targetFitness) {
                        newPopulation.add(trial);
                        
                        if (trialFitness < bestFitness) {
                            bestFitness = trialFitness;
                            bestSolution = trial.clone();
                        }
                    } else {
                        newPopulation.add(target);
                        
                        if (targetFitness < bestFitness) {
                            bestFitness = targetFitness;
                            bestSolution = target.clone();
                        }
                    }
                }
                
                population = newPopulation;
                
                if (generation % 100 == 0) {
                    System.out.printf("Generation %d: Best fitness = %.6f\n", generation, bestFitness);
                }
            }
            
            return new OptimizationResult(bestSolution, bestFitness, OptimizationStatus.OPTIMAL);
        }
        
        private List<double[]> initializePopulation() {
            List<double[]> population = new ArrayList<>();
            
            for (int i = 0; i < populationSize; i++) {
                double[] individual = new double[lowerBounds.length];
                for (int j = 0; j < individual.length; j++) {
                    individual[j] = lowerBounds[j] + random.nextDouble() * (upperBounds[j] - lowerBounds[j]);
                }
                population.add(individual);
            }
            
            return population;
        }
        
        private double[] createTrialVector(List<double[]> population, int targetIndex) {
            int dimensions = lowerBounds.length;
            
            // Select three random individuals (different from target)
            int[] indices = new int[3];
            for (int i = 0; i < 3; i++) {
                do {
                    indices[i] = random.nextInt(populationSize);
                } while (indices[i] == targetIndex || contains(indices, indices[i], i));
            }
            
            double[] mutant = new double[dimensions];
            for (int i = 0; i < dimensions; i++) {
                mutant[i] = population.get(indices[0])[i] + 
                           mutationFactor * (population.get(indices[1])[i] - population.get(indices[2])[i]);
                
                // Ensure bounds
                mutant[i] = Math.max(lowerBounds[i], Math.min(upperBounds[i], mutant[i]));
            }
            
            // Crossover
            double[] trial = population.get(targetIndex).clone();
            int randomIndex = random.nextInt(dimensions); // Ensure at least one gene from mutant
            
            for (int i = 0; i < dimensions; i++) {
                if (random.nextDouble() < crossoverRate || i == randomIndex) {
                    trial[i] = mutant[i];
                }
            }
            
            return trial;
        }
        
        private boolean contains(int[] array, int value, int upTo) {
            for (int i = 0; i < upTo; i++) {
                if (array[i] == value) return true;
            }
            return false;
        }
    }
    
    /**
     * Constrained Optimization using Penalty Methods
     */
    public static class PenaltyMethod {
        private Function<double[], Double> objectiveFunction;
        private List<Function<double[], Double>> equalityConstraints;
        private List<Function<double[], Double>> inequalityConstraints;
        private double penaltyParameter;
        private double penaltyGrowthFactor;
        
        public PenaltyMethod(Function<double[], Double> objectiveFunction) {
            this.objectiveFunction = objectiveFunction;
            this.equalityConstraints = new ArrayList<>();
            this.inequalityConstraints = new ArrayList<>();
            this.penaltyParameter = 1.0;
            this.penaltyGrowthFactor = 10.0;
        }
        
        public void addEqualityConstraint(Function<double[], Double> constraint) {
            equalityConstraints.add(constraint);
        }
        
        public void addInequalityConstraint(Function<double[], Double> constraint) {
            inequalityConstraints.add(constraint);
        }
        
        public OptimizationResult optimize(double[] initialSolution, int maxOuterIterations) {
            double[] x = initialSolution.clone();
            
            for (int outerIter = 0; outerIter < maxOuterIterations; outerIter++) {
                // Create penalized objective function
                Function<double[], Double> penalizedObjective = createPenalizedObjective();
                
                // Solve unconstrained problem
                GradientDescent gd = new GradientDescent(
                    penalizedObjective,
                    GradientDescent.numericalGradient(penalizedObjective, 1e-6),
                    0.01, 1000, 1e-6
                );
                
                OptimizationResult result = gd.optimize(x);
                x = result.solution;
                
                // Check constraint violation
                double maxViolation = 0;
                for (Function<double[], Double> eq : equalityConstraints) {
                    maxViolation = Math.max(maxViolation, Math.abs(eq.apply(x)));
                }
                for (Function<double[], Double> ineq : inequalityConstraints) {
                    maxViolation = Math.max(maxViolation, Math.max(0, ineq.apply(x)));
                }
                
                System.out.printf("Outer iteration %d: f(x) = %.6f, max violation = %.6f\n",
                                 outerIter, objectiveFunction.apply(x), maxViolation);
                
                if (maxViolation < 1e-6) {
                    return new OptimizationResult(x, objectiveFunction.apply(x), OptimizationStatus.OPTIMAL);
                }
                
                // Increase penalty parameter
                penaltyParameter *= penaltyGrowthFactor;
            }
            
            return new OptimizationResult(x, objectiveFunction.apply(x), OptimizationStatus.MAX_ITERATIONS);
        }
        
        private Function<double[], Double> createPenalizedObjective() {
            return x -> {
                double objective = objectiveFunction.apply(x);
                double penalty = 0;
                
                // Equality constraints penalty
                for (Function<double[], Double> eq : equalityConstraints) {
                    double violation = eq.apply(x);
                    penalty += penaltyParameter * violation * violation;
                }
                
                // Inequality constraints penalty
                for (Function<double[], Double> ineq : inequalityConstraints) {
                    double violation = Math.max(0, ineq.apply(x));
                    penalty += penaltyParameter * violation * violation;
                }
                
                return objective + penalty;
            };
        }
    }
    
    // Utility classes
    public static class OptimizationResult {
        public final double[] solution;
        public final double objectiveValue;
        public final OptimizationStatus status;
        
        public OptimizationResult(double[] solution, double objectiveValue, OptimizationStatus status) {
            this.solution = solution;
            this.objectiveValue = objectiveValue;
            this.status = status;
        }
        
        @Override
        public String toString() {
            return String.format("Status: %s, Objective: %.6f, Solution: %s", 
                               status, objectiveValue, Arrays.toString(solution));
        }
    }
    
    public enum OptimizationStatus {
        OPTIMAL, UNBOUNDED, INFEASIBLE, MAX_ITERATIONS
    }
    
    public static void main(String[] args) {
        System.out.println("Optimization Algorithms Demo:");
        System.out.println("=============================");
        
        // Linear Programming Demo
        System.out.println("1. Linear Programming (Simplex Method):");
        
        // Maximize: 3x1 + 2x2
        // Subject to: x1 + x2 <= 4
        //            2x1 + x2 <= 6
        //            x1, x2 >= 0
        
        double[] objective = {3, 2};
        double[][] constraints = {{1, 1}, {2, 1}};
        double[] rhs = {4, 6};
        
        SimplexMethod simplex = new SimplexMethod(objective, constraints, rhs, true);
        OptimizationResult lpResult = simplex.solve();
        System.out.println("Linear Programming Result: " + lpResult);
        
        // Genetic Algorithm Demo
        System.out.println("\n2. Genetic Algorithm:");
        
        // Minimize: x^2 + y^2 (should find (0,0))
        Function<double[], Double> sphere = x -> x[0]*x[0] + x[1]*x[1];
        double[] lowerBounds = {-5, -5};
        double[] upperBounds = {5, 5};
        
        GeneticAlgorithm ga = new GeneticAlgorithm(sphere, lowerBounds, upperBounds, 50, 0.1, 0.8, 500);
        OptimizationResult gaResult = ga.optimize();
        System.out.println("Genetic Algorithm Result: " + gaResult);
        
        // Simulated Annealing Demo
        System.out.println("\n3. Simulated Annealing:");
        
        // Rastrigin function (multimodal)
        Function<double[], Double> rastrigin = x -> {
            double sum = 0;
            for (double xi : x) {
                sum += xi*xi - 10*Math.cos(2*Math.PI*xi) + 10;
            }
            return sum;
        };
        
        SimulatedAnnealing sa = new SimulatedAnnealing(rastrigin, lowerBounds, upperBounds, 
                                                      100, 0.001, 0.95, 5000);
        OptimizationResult saResult = sa.optimize(new double[]{3, -2});
        System.out.println("Simulated Annealing Result: " + saResult);
        
        // Particle Swarm Optimization Demo
        System.out.println("\n4. Particle Swarm Optimization:");
        
        // Rosenbrock function
        Function<double[], Double> rosenbrock = x -> {
            double sum = 0;
            for (int i = 0; i < x.length - 1; i++) {
                sum += 100 * Math.pow(x[i+1] - x[i]*x[i], 2) + Math.pow(1 - x[i], 2);
            }
            return sum;
        };
        
        ParticleSwarmOptimization pso = new ParticleSwarmOptimization(rosenbrock, 
                                                                     new double[]{-2, -2}, 
                                                                     new double[]{2, 2}, 30, 1000);
        OptimizationResult psoResult = pso.optimize();
        System.out.println("PSO Result: " + psoResult);
        
        // Gradient Descent Demo
        System.out.println("\n5. Gradient Descent:");
        
        // Quadratic function: f(x,y) = x^2 + 2y^2
        Function<double[], Double> quadratic = x -> x[0]*x[0] + 2*x[1]*x[1];
        Function<double[], double[]> quadraticGrad = x -> new double[]{2*x[0], 4*x[1]};
        
        GradientDescent gd = new GradientDescent(quadratic, quadraticGrad, 0.1, 1000, 1e-6);
        OptimizationResult gdResult = gd.optimize(new double[]{2, 1});
        System.out.println("Gradient Descent Result: " + gdResult);
        
        // Differential Evolution Demo
        System.out.println("\n6. Differential Evolution:");
        
        // Ackley function
        Function<double[], Double> ackley = x -> {
            double sum1 = 0, sum2 = 0;
            for (double xi : x) {
                sum1 += xi * xi;
                sum2 += Math.cos(2 * Math.PI * xi);
            }
            return -20 * Math.exp(-0.2 * Math.sqrt(sum1 / x.length)) - 
                   Math.exp(sum2 / x.length) + 20 + Math.E;
        };
        
        DifferentialEvolution de = new DifferentialEvolution(ackley, 
                                                            new double[]{-5, -5}, 
                                                            new double[]{5, 5}, 
                                                            30, 0.7, 0.9, 500);
        OptimizationResult deResult = de.optimize();
        System.out.println("Differential Evolution Result: " + deResult);
        
        // Constrained Optimization Demo
        System.out.println("\n7. Constrained Optimization (Penalty Method):");
        
        // Minimize: (x-1)^2 + (y-1)^2
        // Subject to: x + y = 1
        Function<double[], Double> constrainedObj = x -> Math.pow(x[0]-1, 2) + Math.pow(x[1]-1, 2);
        
        PenaltyMethod penalty = new PenaltyMethod(constrainedObj);
        penalty.addEqualityConstraint(x -> x[0] + x[1] - 1); // x + y = 1
        
        OptimizationResult penaltyResult = penalty.optimize(new double[]{0, 0}, 10);
        System.out.println("Constrained Optimization Result: " + penaltyResult);
        
        System.out.println("\nOptimization algorithms demonstration completed!");
    }
}
