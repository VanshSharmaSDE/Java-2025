package Algorithms.QuantumComputing;

/**
 * Quantum Computing Algorithms and Simulations
 * Classical implementations of quantum algorithms for educational purposes
 */
public class QuantumAlgorithms {
    
    /**
     * Complex number representation for quantum states
     */
    public static class Complex {
        public double real, imag;
        
        public Complex(double real, double imag) {
            this.real = real;
            this.imag = imag;
        }
        
        public Complex add(Complex other) {
            return new Complex(real + other.real, imag + other.imag);
        }
        
        public Complex multiply(Complex other) {
            return new Complex(real * other.real - imag * other.imag,
                             real * other.imag + imag * other.real);
        }
        
        public Complex conjugate() {
            return new Complex(real, -imag);
        }
        
        public double magnitude() {
            return Math.sqrt(real * real + imag * imag);
        }
        
        public double phase() {
            return Math.atan2(imag, real);
        }
        
        @Override
        public String toString() {
            if (imag >= 0) {
                return String.format("%.3f + %.3fi", real, imag);
            } else {
                return String.format("%.3f - %.3fi", real, -imag);
            }
        }
    }
    
    /**
     * Quantum State representation
     */
    public static class QuantumState {
        private Complex[] amplitudes;
        private int numQubits;
        
        public QuantumState(int numQubits) {
            this.numQubits = numQubits;
            this.amplitudes = new Complex[1 << numQubits];
            
            // Initialize to |0...0⟩ state
            for (int i = 0; i < amplitudes.length; i++) {
                amplitudes[i] = new Complex(0, 0);
            }
            amplitudes[0] = new Complex(1, 0);
        }
        
        public QuantumState(Complex[] amplitudes) {
            this.amplitudes = amplitudes.clone();
            this.numQubits = (int) (Math.log(amplitudes.length) / Math.log(2));
        }
        
        public Complex getAmplitude(int state) {
            return amplitudes[state];
        }
        
        public void setAmplitude(int state, Complex amplitude) {
            amplitudes[state] = amplitude;
        }
        
        public double getProbability(int state) {
            return amplitudes[state].magnitude() * amplitudes[state].magnitude();
        }
        
        public int getNumQubits() {
            return numQubits;
        }
        
        public int getNumStates() {
            return amplitudes.length;
        }
        
        /**
         * Normalize the quantum state
         */
        public void normalize() {
            double sum = 0;
            for (Complex amp : amplitudes) {
                sum += amp.magnitude() * amp.magnitude();
            }
            
            double norm = Math.sqrt(sum);
            for (int i = 0; i < amplitudes.length; i++) {
                amplitudes[i] = new Complex(amplitudes[i].real / norm, amplitudes[i].imag / norm);
            }
        }
        
        /**
         * Measure the quantum state (collapses to classical state)
         */
        public int measure() {
            double rand = Math.random();
            double cumulative = 0;
            
            for (int i = 0; i < amplitudes.length; i++) {
                cumulative += getProbability(i);
                if (rand <= cumulative) {
                    // Collapse to this state
                    for (int j = 0; j < amplitudes.length; j++) {
                        amplitudes[j] = new Complex(0, 0);
                    }
                    amplitudes[i] = new Complex(1, 0);
                    return i;
                }
            }
            
            return amplitudes.length - 1; // Fallback
        }
        
        public void display() {
            System.out.println("Quantum State:");
            for (int i = 0; i < amplitudes.length; i++) {
                if (amplitudes[i].magnitude() > 1e-10) {
                    String binary = String.format("%" + numQubits + "s", 
                                                 Integer.toBinaryString(i)).replace(' ', '0');
                    System.out.printf("|%s⟩: %s (prob: %.3f)\n", 
                                    binary, amplitudes[i], getProbability(i));
                }
            }
        }
    }
    
    /**
     * Quantum Gates
     */
    public static class QuantumGates {
        
        /**
         * Pauli-X Gate (NOT gate)
         */
        public static void pauliX(QuantumState state, int qubit) {
            int mask = 1 << qubit;
            Complex[] newAmplitudes = new Complex[state.getNumStates()];
            
            for (int i = 0; i < state.getNumStates(); i++) {
                int flipped = i ^ mask;
                newAmplitudes[flipped] = state.getAmplitude(i);
            }
            
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, newAmplitudes[i]);
            }
        }
        
        /**
         * Hadamard Gate (creates superposition)
         */
        public static void hadamard(QuantumState state, int qubit) {
            int mask = 1 << qubit;
            Complex[] newAmplitudes = new Complex[state.getNumStates()];
            
            for (int i = 0; i < state.getNumStates(); i++) {
                newAmplitudes[i] = new Complex(0, 0);
            }
            
            double sqrt2 = Math.sqrt(2);
            
            for (int i = 0; i < state.getNumStates(); i++) {
                Complex amp = state.getAmplitude(i);
                int flipped = i ^ mask;
                
                // |0⟩ component
                newAmplitudes[i] = newAmplitudes[i].add(
                    new Complex(amp.real / sqrt2, amp.imag / sqrt2));
                
                // |1⟩ component (with phase)
                if ((i & mask) == 0) {
                    newAmplitudes[flipped] = newAmplitudes[flipped].add(
                        new Complex(amp.real / sqrt2, amp.imag / sqrt2));
                } else {
                    newAmplitudes[flipped] = newAmplitudes[flipped].add(
                        new Complex(-amp.real / sqrt2, -amp.imag / sqrt2));
                }
            }
            
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, newAmplitudes[i]);
            }
        }
        
        /**
         * Controlled-NOT (CNOT) Gate
         */
        public static void cnot(QuantumState state, int control, int target) {
            int controlMask = 1 << control;
            int targetMask = 1 << target;
            Complex[] newAmplitudes = new Complex[state.getNumStates()];
            
            for (int i = 0; i < state.getNumStates(); i++) {
                if ((i & controlMask) != 0) {
                    // Control qubit is 1, flip target
                    newAmplitudes[i ^ targetMask] = state.getAmplitude(i);
                } else {
                    // Control qubit is 0, no change
                    newAmplitudes[i] = state.getAmplitude(i);
                }
            }
            
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, newAmplitudes[i]);
            }
        }
        
        /**
         * Phase Gate
         */
        public static void phase(QuantumState state, int qubit, double theta) {
            int mask = 1 << qubit;
            Complex phaseShift = new Complex(Math.cos(theta), Math.sin(theta));
            
            for (int i = 0; i < state.getNumStates(); i++) {
                if ((i & mask) != 0) {
                    state.setAmplitude(i, state.getAmplitude(i).multiply(phaseShift));
                }
            }
        }
        
        /**
         * Rotation around Y-axis
         */
        public static void rotationY(QuantumState state, int qubit, double theta) {
            int mask = 1 << qubit;
            Complex[] newAmplitudes = new Complex[state.getNumStates()];
            
            for (int i = 0; i < state.getNumStates(); i++) {
                newAmplitudes[i] = new Complex(0, 0);
            }
            
            double cosHalf = Math.cos(theta / 2);
            double sinHalf = Math.sin(theta / 2);
            
            for (int i = 0; i < state.getNumStates(); i++) {
                Complex amp = state.getAmplitude(i);
                int flipped = i ^ mask;
                
                if ((i & mask) == 0) {
                    // |0⟩ state
                    newAmplitudes[i] = newAmplitudes[i].add(
                        new Complex(amp.real * cosHalf, amp.imag * cosHalf));
                    newAmplitudes[flipped] = newAmplitudes[flipped].add(
                        new Complex(amp.real * sinHalf, amp.imag * sinHalf));
                } else {
                    // |1⟩ state
                    newAmplitudes[i] = newAmplitudes[i].add(
                        new Complex(amp.real * cosHalf, amp.imag * cosHalf));
                    newAmplitudes[flipped] = newAmplitudes[flipped].add(
                        new Complex(-amp.real * sinHalf, -amp.imag * sinHalf));
                }
            }
            
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, newAmplitudes[i]);
            }
        }
    }
    
    /**
     * Quantum Algorithms
     */
    public static class Algorithms {
        
        /**
         * Deutsch's Algorithm
         * Determines if a function is constant or balanced
         */
        public static boolean deutschAlgorithm(java.util.function.Function<Integer, Integer> oracle) {
            QuantumState state = new QuantumState(2);
            
            // Initialize ancilla qubit to |1⟩
            QuantumGates.pauliX(state, 1);
            
            // Apply Hadamard to both qubits
            QuantumGates.hadamard(state, 0);
            QuantumGates.hadamard(state, 1);
            
            // Apply oracle (simplified classical simulation)
            applyOracle(state, oracle);
            
            // Apply Hadamard to first qubit
            QuantumGates.hadamard(state, 0);
            
            // Measure first qubit
            int result = measureQubit(state, 0);
            
            return result == 0; // 0 = constant, 1 = balanced
        }
        
        /**
         * Grover's Algorithm (simplified version)
         * Searches for a marked item in unsorted database
         */
        public static int groversAlgorithm(int numItems, int markedItem) {
            int numQubits = (int) Math.ceil(Math.log(numItems) / Math.log(2));
            QuantumState state = new QuantumState(numQubits);
            
            // Initialize to equal superposition
            for (int i = 0; i < numQubits; i++) {
                QuantumGates.hadamard(state, i);
            }
            
            int iterations = (int) Math.round(Math.PI * Math.sqrt(numItems) / 4);
            
            for (int iter = 0; iter < iterations; iter++) {
                // Oracle: flip phase of marked item
                groversOracle(state, markedItem);
                
                // Diffusion operator
                groversDiffusion(state);
            }
            
            return state.measure();
        }
        
        private static void groversOracle(QuantumState state, int markedItem) {
            if (markedItem < state.getNumStates()) {
                Complex amp = state.getAmplitude(markedItem);
                state.setAmplitude(markedItem, new Complex(-amp.real, -amp.imag));
            }
        }
        
        private static void groversDiffusion(QuantumState state) {
            // Simplified diffusion operator
            double avgReal = 0, avgImag = 0;
            for (int i = 0; i < state.getNumStates(); i++) {
                Complex amp = state.getAmplitude(i);
                avgReal += amp.real;
                avgImag += amp.imag;
            }
            avgReal /= state.getNumStates();
            avgImag /= state.getNumStates();
            
            for (int i = 0; i < state.getNumStates(); i++) {
                Complex amp = state.getAmplitude(i);
                state.setAmplitude(i, new Complex(2 * avgReal - amp.real, 2 * avgImag - amp.imag));
            }
        }
        
        /**
         * Quantum Fourier Transform (simplified)
         */
        public static void quantumFourierTransform(QuantumState state) {
            int n = state.getNumQubits();
            
            for (int i = 0; i < n; i++) {
                QuantumGates.hadamard(state, i);
                
                for (int j = i + 1; j < n; j++) {
                    double theta = Math.PI / Math.pow(2, j - i);
                    controlledPhase(state, j, i, theta);
                }
            }
            
            // Reverse the order of qubits
            for (int i = 0; i < n / 2; i++) {
                swapQubits(state, i, n - 1 - i);
            }
        }
        
        private static void controlledPhase(QuantumState state, int control, int target, double theta) {
            int controlMask = 1 << control;
            int targetMask = 1 << target;
            Complex phaseShift = new Complex(Math.cos(theta), Math.sin(theta));
            
            for (int i = 0; i < state.getNumStates(); i++) {
                if ((i & controlMask) != 0 && (i & targetMask) != 0) {
                    state.setAmplitude(i, state.getAmplitude(i).multiply(phaseShift));
                }
            }
        }
        
        private static void swapQubits(QuantumState state, int qubit1, int qubit2) {
            int mask1 = 1 << qubit1;
            int mask2 = 1 << qubit2;
            Complex[] newAmplitudes = new Complex[state.getNumStates()];
            
            for (int i = 0; i < state.getNumStates(); i++) {
                int bit1 = (i & mask1) >> qubit1;
                int bit2 = (i & mask2) >> qubit2;
                
                int newState = i;
                newState = (newState & ~mask1) | (bit2 << qubit1);
                newState = (newState & ~mask2) | (bit1 << qubit2);
                
                newAmplitudes[newState] = state.getAmplitude(i);
            }
            
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, newAmplitudes[i]);
            }
        }
        
        /**
         * Shor's Algorithm (classical part - period finding)
         */
        public static int shorsAlgorithm(int N) {
            // Classical preprocessing
            if (N % 2 == 0) return 2;
            
            // Find a random number a < N
            int a = 2 + (int) (Math.random() * (N - 2));
            
            // Find GCD(a, N)
            int gcd = gcd(a, N);
            if (gcd > 1) return gcd;
            
            // Quantum period finding (simplified classical simulation)
            int period = findPeriod(a, N);
            
            if (period % 2 != 0) {
                return shorsAlgorithm(N); // Try again
            }
            
            int factor1 = gcd((int) Math.pow(a, period / 2) - 1, N);
            int factor2 = gcd((int) Math.pow(a, period / 2) + 1, N);
            
            if (factor1 > 1 && factor1 < N) return factor1;
            if (factor2 > 1 && factor2 < N) return factor2;
            
            return shorsAlgorithm(N); // Try again
        }
        
        private static int findPeriod(int a, int N) {
            // Simplified classical period finding
            int period = 1;
            int current = a % N;
            
            while (current != 1 && period < N) {
                current = (current * a) % N;
                period++;
            }
            
            return period;
        }
        
        private static int gcd(int a, int b) {
            while (b != 0) {
                int temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }
        
        // Helper methods
        private static void applyOracle(QuantumState state, java.util.function.Function<Integer, Integer> oracle) {
            // Simplified oracle application
            for (int i = 0; i < state.getNumStates(); i++) {
                int input = i >> 1; // First qubit
                int output = oracle.apply(input);
                if (output == 1) {
                    // Flip the ancilla qubit
                    int flipped = i ^ 1;
                    Complex temp = state.getAmplitude(i);
                    state.setAmplitude(i, state.getAmplitude(flipped));
                    state.setAmplitude(flipped, temp);
                }
            }
        }
        
        private static int measureQubit(QuantumState state, int qubit) {
            int mask = 1 << qubit;
            double prob0 = 0;
            
            for (int i = 0; i < state.getNumStates(); i++) {
                if ((i & mask) == 0) {
                    prob0 += state.getProbability(i);
                }
            }
            
            return Math.random() < prob0 ? 0 : 1;
        }
    }
    
    /**
     * Quantum Error Correction
     */
    public static class ErrorCorrection {
        
        /**
         * Three-qubit bit flip code
         */
        public static void threeQubitBitFlipCode() {
            System.out.println("Three-Qubit Bit Flip Code:");
            
            QuantumState state = new QuantumState(3);
            
            // Encode logical |0⟩ as |000⟩
            System.out.println("Encoded |0⟩:");
            state.display();
            
            // Introduce bit flip error on qubit 1
            QuantumGates.pauliX(state, 1);
            System.out.println("\nAfter bit flip error on qubit 1:");
            state.display();
            
            // Error correction (majority vote)
            correctBitFlipError(state);
            System.out.println("\nAfter error correction:");
            state.display();
        }
        
        private static void correctBitFlipError(QuantumState state) {
            // Simplified error correction
            int measurement = state.measure();
            String binary = String.format("%3s", Integer.toBinaryString(measurement)).replace(' ', '0');
            
            // Count 1s and 0s
            int ones = 0;
            for (char c : binary.toCharArray()) {
                if (c == '1') ones++;
            }
            
            // Apply majority vote
            for (int i = 0; i < state.getNumStates(); i++) {
                state.setAmplitude(i, new Complex(0, 0));
            }
            
            if (ones >= 2) {
                state.setAmplitude(7, new Complex(1, 0)); // |111⟩
            } else {
                state.setAmplitude(0, new Complex(1, 0)); // |000⟩
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Quantum Computing Algorithms Demo:");
        System.out.println("===================================");
        
        // Basic quantum state manipulation
        System.out.println("1. Basic Quantum State:");
        QuantumState state = new QuantumState(2);
        state.display();
        
        System.out.println("\nAfter Hadamard on qubit 0:");
        QuantumGates.hadamard(state, 0);
        state.display();
        
        System.out.println("\nAfter CNOT (0 -> 1):");
        QuantumGates.cnot(state, 0, 1);
        state.display();
        
        // Deutsch's Algorithm
        System.out.println("\n2. Deutsch's Algorithm:");
        boolean isConstant = Algorithms.deutschAlgorithm(x -> 0); // Constant function
        System.out.println("Constant function result: " + (isConstant ? "Constant" : "Balanced"));
        
        boolean isBalanced = Algorithms.deutschAlgorithm(x -> x); // Balanced function
        System.out.println("Balanced function result: " + (isBalanced ? "Constant" : "Balanced"));
        
        // Grover's Algorithm
        System.out.println("\n3. Grover's Algorithm:");
        int numItems = 8;
        int markedItem = 5;
        int found = Algorithms.groversAlgorithm(numItems, markedItem);
        System.out.println("Searching for item " + markedItem + " in " + numItems + " items");
        System.out.println("Found item: " + found);
        
        // Quantum Fourier Transform
        System.out.println("\n4. Quantum Fourier Transform:");
        QuantumState qftState = new QuantumState(3);
        QuantumGates.pauliX(qftState, 0); // Set to |001⟩
        System.out.println("Before QFT:");
        qftState.display();
        
        Algorithms.quantumFourierTransform(qftState);
        System.out.println("After QFT:");
        qftState.display();
        
        // Shor's Algorithm
        System.out.println("\n5. Shor's Algorithm (Classical Simulation):");
        int N = 15;
        int factor = Algorithms.shorsAlgorithm(N);
        System.out.println("Factor of " + N + ": " + factor);
        
        // Error Correction
        System.out.println("\n6. Quantum Error Correction:");
        ErrorCorrection.threeQubitBitFlipCode();
        
        // Quantum Entanglement demonstration
        System.out.println("\n7. Bell State (Maximum Entanglement):");
        QuantumState bellState = new QuantumState(2);
        QuantumGates.hadamard(bellState, 0);
        QuantumGates.cnot(bellState, 0, 1);
        bellState.display();
        
        System.out.println("Measuring Bell state:");
        int result = bellState.measure();
        System.out.println("Measured state: |" + 
                          String.format("%2s", Integer.toBinaryString(result)).replace(' ', '0') + "⟩");
    }
}
