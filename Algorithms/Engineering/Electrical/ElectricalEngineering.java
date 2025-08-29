package Algorithms.Engineering.Electrical;

/**
 * Electrical Engineering Algorithms and Formulas
 */
public class ElectricalEngineering {
    
    // Physical constants
    private static final double ELECTRON_CHARGE = 1.602e-19; // Coulombs
    private static final double PERMITTIVITY_FREE_SPACE = 8.854e-12; // F/m
    private static final double PERMEABILITY_FREE_SPACE = 4 * Math.PI * 1e-7; // H/m
    private static final double SPEED_OF_LIGHT = 299792458; // m/s
    
    /**
     * Ohm's Law calculations
     */
    public static class OhmsLaw {
        
        /**
         * Calculate voltage: V = I × R
         * @param current Current in Amperes
         * @param resistance Resistance in Ohms
         * @return Voltage in Volts
         */
        public static double calculateVoltage(double current, double resistance) {
            return current * resistance;
        }
        
        /**
         * Calculate current: I = V / R
         * @param voltage Voltage in Volts
         * @param resistance Resistance in Ohms
         * @return Current in Amperes
         */
        public static double calculateCurrent(double voltage, double resistance) {
            return voltage / resistance;
        }
        
        /**
         * Calculate resistance: R = V / I
         * @param voltage Voltage in Volts
         * @param current Current in Amperes
         * @return Resistance in Ohms
         */
        public static double calculateResistance(double voltage, double current) {
            return voltage / current;
        }
        
        /**
         * Calculate power: P = V × I = I²R = V²/R
         * @param voltage Voltage in Volts
         * @param current Current in Amperes
         * @return Power in Watts
         */
        public static double calculatePower(double voltage, double current) {
            return voltage * current;
        }
        
        /**
         * Calculate power using current and resistance: P = I²R
         * @param current Current in Amperes
         * @param resistance Resistance in Ohms
         * @return Power in Watts
         */
        public static double calculatePowerIR(double current, double resistance) {
            return current * current * resistance;
        }
        
        /**
         * Calculate power using voltage and resistance: P = V²/R
         * @param voltage Voltage in Volts
         * @param resistance Resistance in Ohms
         * @return Power in Watts
         */
        public static double calculatePowerVR(double voltage, double resistance) {
            return (voltage * voltage) / resistance;
        }
    }
    
    /**
     * AC Circuit Analysis
     */
    public static class ACCircuits {
        
        /**
         * Calculate capacitive reactance: Xc = 1/(2πfC)
         * @param frequency Frequency in Hz
         * @param capacitance Capacitance in Farads
         * @return Capacitive reactance in Ohms
         */
        public static double capacitiveReactance(double frequency, double capacitance) {
            return 1.0 / (2 * Math.PI * frequency * capacitance);
        }
        
        /**
         * Calculate inductive reactance: Xl = 2πfL
         * @param frequency Frequency in Hz
         * @param inductance Inductance in Henries
         * @return Inductive reactance in Ohms
         */
        public static double inductiveReactance(double frequency, double inductance) {
            return 2 * Math.PI * frequency * inductance;
        }
        
        /**
         * Calculate impedance magnitude for RLC circuit: |Z| = √(R² + (Xl - Xc)²)
         * @param resistance Resistance in Ohms
         * @param inductiveReactance Inductive reactance in Ohms
         * @param capacitiveReactance Capacitive reactance in Ohms
         * @return Impedance magnitude in Ohms
         */
        public static double impedanceMagnitude(double resistance, double inductiveReactance, 
                                              double capacitiveReactance) {
            double reactance = inductiveReactance - capacitiveReactance;
            return Math.sqrt(resistance * resistance + reactance * reactance);
        }
        
        /**
         * Calculate phase angle for RLC circuit: φ = arctan((Xl - Xc)/R)
         * @param resistance Resistance in Ohms
         * @param inductiveReactance Inductive reactance in Ohms
         * @param capacitiveReactance Capacitive reactance in Ohms
         * @return Phase angle in radians
         */
        public static double phaseAngle(double resistance, double inductiveReactance, 
                                      double capacitiveReactance) {
            double reactance = inductiveReactance - capacitiveReactance;
            return Math.atan(reactance / resistance);
        }
        
        /**
         * Calculate resonant frequency: f₀ = 1/(2π√(LC))
         * @param inductance Inductance in Henries
         * @param capacitance Capacitance in Farads
         * @return Resonant frequency in Hz
         */
        public static double resonantFrequency(double inductance, double capacitance) {
            return 1.0 / (2 * Math.PI * Math.sqrt(inductance * capacitance));
        }
        
        /**
         * Calculate RMS value from peak value: RMS = Peak/√2
         * @param peakValue Peak value
         * @return RMS value
         */
        public static double rmsValue(double peakValue) {
            return peakValue / Math.sqrt(2);
        }
        
        /**
         * Calculate average power in AC circuit: P = VrmsIrmscosφ
         * @param voltageRMS RMS voltage in Volts
         * @param currentRMS RMS current in Amperes
         * @param phaseAngle Phase angle in radians
         * @return Average power in Watts
         */
        public static double averagePower(double voltageRMS, double currentRMS, double phaseAngle) {
            return voltageRMS * currentRMS * Math.cos(phaseAngle);
        }
    }
    
    /**
     * Electromagnetic Field Calculations
     */
    public static class ElectromagneticFields {
        
        /**
         * Calculate electric field strength: E = F/q
         * @param force Force in Newtons
         * @param charge Charge in Coulombs
         * @return Electric field strength in N/C or V/m
         */
        public static double electricFieldStrength(double force, double charge) {
            return force / charge;
        }
        
        /**
         * Calculate electric field due to point charge: E = kq/r²
         * @param charge Charge in Coulombs
         * @param distance Distance in meters
         * @return Electric field strength in N/C
         */
        public static double electricFieldPointCharge(double charge, double distance) {
            double k = 1.0 / (4 * Math.PI * PERMITTIVITY_FREE_SPACE);
            return k * charge / (distance * distance);
        }
        
        /**
         * Calculate capacitance of parallel plate capacitor: C = ε₀A/d
         * @param area Plate area in m²
         * @param distance Distance between plates in meters
         * @param relativePermittivity Relative permittivity of dielectric
         * @return Capacitance in Farads
         */
        public static double parallelPlateCapacitance(double area, double distance, 
                                                    double relativePermittivity) {
            return relativePermittivity * PERMITTIVITY_FREE_SPACE * area / distance;
        }
        
        /**
         * Calculate inductance of solenoid: L = μ₀n²Al
         * @param turns Number of turns
         * @param area Cross-sectional area in m²
         * @param length Length in meters
         * @return Inductance in Henries
         */
        public static double solenoidInductance(int turns, double area, double length) {
            double n = turns / length; // Turns per unit length
            return PERMEABILITY_FREE_SPACE * n * n * area * length;
        }
        
        /**
         * Calculate magnetic field inside solenoid: B = μ₀nI
         * @param turns Number of turns
         * @param length Length in meters
         * @param current Current in Amperes
         * @return Magnetic field in Tesla
         */
        public static double solenoidMagneticField(int turns, double length, double current) {
            double n = turns / length;
            return PERMEABILITY_FREE_SPACE * n * current;
        }
        
        /**
         * Calculate energy stored in capacitor: U = ½CV²
         * @param capacitance Capacitance in Farads
         * @param voltage Voltage in Volts
         * @return Energy in Joules
         */
        public static double capacitorEnergy(double capacitance, double voltage) {
            return 0.5 * capacitance * voltage * voltage;
        }
        
        /**
         * Calculate energy stored in inductor: U = ½LI²
         * @param inductance Inductance in Henries
         * @param current Current in Amperes
         * @return Energy in Joules
         */
        public static double inductorEnergy(double inductance, double current) {
            return 0.5 * inductance * current * current;
        }
    }
    
    /**
     * Power System Calculations
     */
    public static class PowerSystems {
        
        /**
         * Calculate three-phase power: P = √3 × Vl × Il × cosφ
         * @param lineVoltage Line voltage in Volts
         * @param lineCurrent Line current in Amperes
         * @param powerFactor Power factor (cosφ)
         * @return Three-phase power in Watts
         */
        public static double threePhaseActivePower(double lineVoltage, double lineCurrent, 
                                                 double powerFactor) {
            return Math.sqrt(3) * lineVoltage * lineCurrent * powerFactor;
        }
        
        /**
         * Calculate apparent power: S = √3 × Vl × Il
         * @param lineVoltage Line voltage in Volts
         * @param lineCurrent Line current in Amperes
         * @return Apparent power in VA
         */
        public static double threePhaseApparentPower(double lineVoltage, double lineCurrent) {
            return Math.sqrt(3) * lineVoltage * lineCurrent;
        }
        
        /**
         * Calculate reactive power: Q = √3 × Vl × Il × sinφ
         * @param lineVoltage Line voltage in Volts
         * @param lineCurrent Line current in Amperes
         * @param phaseAngle Phase angle in radians
         * @return Reactive power in VAR
         */
        public static double threePhaseReactivePower(double lineVoltage, double lineCurrent, 
                                                   double phaseAngle) {
            return Math.sqrt(3) * lineVoltage * lineCurrent * Math.sin(phaseAngle);
        }
        
        /**
         * Calculate efficiency: η = Pout/Pin × 100%
         * @param outputPower Output power in Watts
         * @param inputPower Input power in Watts
         * @return Efficiency as percentage
         */
        public static double efficiency(double outputPower, double inputPower) {
            return (outputPower / inputPower) * 100;
        }
        
        /**
         * Calculate power factor from active and apparent power: pf = P/S
         * @param activePower Active power in Watts
         * @param apparentPower Apparent power in VA
         * @return Power factor
         */
        public static double powerFactor(double activePower, double apparentPower) {
            return activePower / apparentPower;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Electrical Engineering Calculations:");
        System.out.println("====================================");
        
        // Ohm's Law examples
        System.out.println("Ohm's Law:");
        double voltage = 12.0; // V
        double resistance = 4.0; // Ω
        double current = OhmsLaw.calculateCurrent(voltage, resistance);
        double power = OhmsLaw.calculatePower(voltage, current);
        
        System.out.println("Voltage: " + voltage + " V");
        System.out.println("Resistance: " + resistance + " Ω");
        System.out.println("Current: " + current + " A");
        System.out.println("Power: " + power + " W");
        
        // AC Circuit analysis
        System.out.println("\nAC Circuit Analysis:");
        double frequency = 60; // Hz
        double capacitance = 100e-6; // 100 μF
        double inductance = 0.1; // H
        double acResistance = 10; // Ω
        
        double xc = ACCircuits.capacitiveReactance(frequency, capacitance);
        double xl = ACCircuits.inductiveReactance(frequency, inductance);
        double impedance = ACCircuits.impedanceMagnitude(acResistance, xl, xc);
        double phase = ACCircuits.phaseAngle(acResistance, xl, xc);
        double resonantFreq = ACCircuits.resonantFrequency(inductance, capacitance);
        
        System.out.println("Frequency: " + frequency + " Hz");
        System.out.println("Capacitive Reactance: " + xc + " Ω");
        System.out.println("Inductive Reactance: " + xl + " Ω");
        System.out.println("Impedance: " + impedance + " Ω");
        System.out.println("Phase Angle: " + Math.toDegrees(phase) + "°");
        System.out.println("Resonant Frequency: " + resonantFreq + " Hz");
        
        // Electromagnetic fields
        System.out.println("\nElectromagnetic Fields:");
        double charge = 1e-6; // 1 μC
        double distance = 0.1; // 10 cm
        double electricField = ElectromagneticFields.electricFieldPointCharge(charge, distance);
        
        double plateArea = 0.01; // 1 cm²
        double plateDistance = 0.001; // 1 mm
        double capacitanceValue = ElectromagneticFields.parallelPlateCapacitance(plateArea, plateDistance, 1.0);
        
        System.out.println("Electric field (point charge): " + electricField + " N/C");
        System.out.println("Parallel plate capacitance: " + capacitanceValue*1e12 + " pF");
        
        // Power systems
        System.out.println("\nThree-Phase Power Systems:");
        double lineVoltage = 400; // V
        double lineCurrent = 10; // A
        double powerFactorValue = 0.8;
        
        double activePower = PowerSystems.threePhaseActivePower(lineVoltage, lineCurrent, powerFactorValue);
        double apparentPower = PowerSystems.threePhaseApparentPower(lineVoltage, lineCurrent);
        double phaseAngle = Math.acos(powerFactorValue);
        double reactivePower = PowerSystems.threePhaseReactivePower(lineVoltage, lineCurrent, phaseAngle);
        
        System.out.println("Line Voltage: " + lineVoltage + " V");
        System.out.println("Line Current: " + lineCurrent + " A");
        System.out.println("Power Factor: " + powerFactorValue);
        System.out.println("Active Power: " + activePower/1000 + " kW");
        System.out.println("Apparent Power: " + apparentPower/1000 + " kVA");
        System.out.println("Reactive Power: " + reactivePower/1000 + " kVAR");
        
        // Efficiency calculation
        double inputPower = 1000; // W
        double outputPower = 850; // W
        double efficiencyValue = PowerSystems.efficiency(outputPower, inputPower);
        System.out.println("Efficiency: " + efficiencyValue + "%");
    }
}
