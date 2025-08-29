package Algorithms.Physics.Relativity;

/**
 * Einstein's Mass-Energy Equivalence: E = mc²
 * Where:
 * E = Energy (Joules)
 * m = Mass (kg)
 * c = Speed of light in vacuum (299,792,458 m/s)
 */
public class MassEnergyEquivalence {
    
    // Speed of light in vacuum (m/s)
    private static final double SPEED_OF_LIGHT = 299792458.0;
    
    /**
     * Calculate energy from mass using E = mc²
     * @param mass Mass in kilograms
     * @return Energy in Joules
     */
    public static double calculateEnergy(double mass) {
        return mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    }
    
    /**
     * Calculate mass from energy using m = E/c²
     * @param energy Energy in Joules
     * @return Mass in kilograms
     */
    public static double calculateMass(double energy) {
        return energy / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
    }
    
    /**
     * Calculate relativistic energy: E = γmc²
     * @param restMass Rest mass in kg
     * @param velocity Velocity in m/s
     * @return Total energy in Joules
     */
    public static double calculateRelativisticEnergy(double restMass, double velocity) {
        double gamma = calculateLorentzFactor(velocity);
        return gamma * restMass * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    }
    
    /**
     * Calculate Lorentz factor: γ = 1/√(1 - v²/c²)
     * @param velocity Velocity in m/s
     * @return Lorentz factor
     */
    public static double calculateLorentzFactor(double velocity) {
        double vSquaredOverCSquared = (velocity * velocity) / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
        return 1.0 / Math.sqrt(1.0 - vSquaredOverCSquared);
    }
    
    public static void main(String[] args) {
        // Example: Calculate energy equivalent of 1 kg of mass
        double mass = 1.0; // kg
        double energy = calculateEnergy(mass);
        
        System.out.println("Mass: " + mass + " kg");
        System.out.println("Energy equivalent: " + energy + " J");
        System.out.println("Energy equivalent: " + (energy / 1e15) + " PJ (petajoules)");
        
        // Example: Calculate mass equivalent of energy
        double inputEnergy = 1e16; // Joules
        double equivalentMass = calculateMass(inputEnergy);
        System.out.println("\nEnergy: " + inputEnergy + " J");
        System.out.println("Mass equivalent: " + equivalentMass + " kg");
        
        // Example: Relativistic energy calculation
        double restMass = 9.109e-31; // Electron rest mass in kg
        double velocity = 0.9 * SPEED_OF_LIGHT; // 90% speed of light
        double relativisticEnergy = calculateRelativisticEnergy(restMass, velocity);
        double lorentzFactor = calculateLorentzFactor(velocity);
        
        System.out.println("\nElectron at 90% speed of light:");
        System.out.println("Rest mass: " + restMass + " kg");
        System.out.println("Velocity: " + velocity + " m/s");
        System.out.println("Lorentz factor: " + lorentzFactor);
        System.out.println("Total energy: " + relativisticEnergy + " J");
    }
}
