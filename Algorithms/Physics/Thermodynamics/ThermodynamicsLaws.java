package Algorithms.Physics.Thermodynamics;

/**
 * Thermodynamics Algorithms and Laws
 */
public class ThermodynamicsLaws {
    
    // Gas constant in J/(mol⋅K)
    private static final double GAS_CONSTANT = 8.314;
    
    // Boltzmann constant in J/K
    private static final double BOLTZMANN_CONSTANT = 1.381e-23;
    
    // Avogadro's number
    private static final double AVOGADRO_NUMBER = 6.022e23;
    
    /**
     * Ideal Gas Law: PV = nRT
     * Calculate pressure given volume, moles, and temperature
     * @param volume Volume in m³
     * @param moles Number of moles
     * @param temperature Temperature in Kelvin
     * @return Pressure in Pascals
     */
    public static double calculatePressure(double volume, double moles, double temperature) {
        return (moles * GAS_CONSTANT * temperature) / volume;
    }
    
    /**
     * Calculate volume using ideal gas law
     * @param pressure Pressure in Pascals
     * @param moles Number of moles
     * @param temperature Temperature in Kelvin
     * @return Volume in m³
     */
    public static double calculateVolume(double pressure, double moles, double temperature) {
        return (moles * GAS_CONSTANT * temperature) / pressure;
    }
    
    /**
     * Calculate temperature using ideal gas law
     * @param pressure Pressure in Pascals
     * @param volume Volume in m³
     * @param moles Number of moles
     * @return Temperature in Kelvin
     */
    public static double calculateTemperature(double pressure, double volume, double moles) {
        return (pressure * volume) / (moles * GAS_CONSTANT);
    }
    
    /**
     * Calculate heat capacity: Q = mcΔT
     * @param mass Mass in kg
     * @param specificHeat Specific heat capacity in J/(kg⋅K)
     * @param temperatureChange Change in temperature in K
     * @return Heat energy in Joules
     */
    public static double calculateHeat(double mass, double specificHeat, double temperatureChange) {
        return mass * specificHeat * temperatureChange;
    }
    
    /**
     * Calculate efficiency of heat engine: η = 1 - (Tc/Th)
     * @param coldTemperature Cold reservoir temperature in Kelvin
     * @param hotTemperature Hot reservoir temperature in Kelvin
     * @return Efficiency (0 to 1)
     */
    public static double calculateCarnotEfficiency(double coldTemperature, double hotTemperature) {
        return 1.0 - (coldTemperature / hotTemperature);
    }
    
    /**
     * Calculate entropy change: ΔS = Q/T
     * @param heat Heat transferred in Joules
     * @param temperature Temperature in Kelvin
     * @return Entropy change in J/K
     */
    public static double calculateEntropyChange(double heat, double temperature) {
        return heat / temperature;
    }
    
    /**
     * Calculate internal energy change (First Law): ΔU = Q - W
     * @param heat Heat added to system in Joules
     * @param work Work done by system in Joules
     * @return Change in internal energy in Joules
     */
    public static double calculateInternalEnergyChange(double heat, double work) {
        return heat - work;
    }
    
    /**
     * Convert Celsius to Kelvin
     * @param celsius Temperature in Celsius
     * @return Temperature in Kelvin
     */
    public static double celsiusToKelvin(double celsius) {
        return celsius + 273.15;
    }
    
    /**
     * Convert Kelvin to Celsius
     * @param kelvin Temperature in Kelvin
     * @return Temperature in Celsius
     */
    public static double kelvinToCelsius(double kelvin) {
        return kelvin - 273.15;
    }
    
    public static void main(String[] args) {
        System.out.println("Thermodynamics Calculations:");
        System.out.println("============================");
        
        // Ideal Gas Law example
        double volume = 0.1; // m³
        double moles = 1.0; // mol
        double temperature = celsiusToKelvin(25); // 25°C to Kelvin
        
        double pressure = calculatePressure(volume, moles, temperature);
        System.out.println("Ideal Gas Law (PV = nRT):");
        System.out.println("Pressure: " + pressure + " Pa");
        
        // Heat calculation example
        double mass = 2.0; // kg (water)
        double specificHeat = 4186; // J/(kg⋅K) for water
        double tempChange = 50; // K
        
        double heat = calculateHeat(mass, specificHeat, tempChange);
        System.out.println("\nHeat Calculation (Q = mcΔT):");
        System.out.println("Heat required: " + heat + " J");
        
        // Carnot efficiency example
        double hotTemp = celsiusToKelvin(500); // 500°C
        double coldTemp = celsiusToKelvin(25);  // 25°C
        
        double efficiency = calculateCarnotEfficiency(coldTemp, hotTemp);
        System.out.println("\nCarnot Efficiency:");
        System.out.println("Efficiency: " + (efficiency * 100) + "%");
        
        // Entropy change example
        double entropyChange = calculateEntropyChange(heat, temperature);
        System.out.println("\nEntropy Change (ΔS = Q/T):");
        System.out.println("Entropy change: " + entropyChange + " J/K");
        
        // First Law of Thermodynamics example
        double workDone = 1000; // J
        double internalEnergyChange = calculateInternalEnergyChange(heat, workDone);
        System.out.println("\nFirst Law (ΔU = Q - W):");
        System.out.println("Internal energy change: " + internalEnergyChange + " J");
    }
}
