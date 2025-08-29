package Algorithms.Engineering.Mechanical;

/**
 * Mechanical Engineering Algorithms
 */
public class MechanicalEngineering {
    
    // Material constants
    private static final double STEEL_DENSITY = 7850; // kg/m³
    private static final double ALUMINUM_DENSITY = 2700; // kg/m³
    private static final double STEEL_YOUNGS_MODULUS = 200e9; // Pa
    
    /**
     * Calculate stress: σ = F/A
     * @param force Applied force in Newtons
     * @param area Cross-sectional area in m²
     * @return Stress in Pascals
     */
    public static double calculateStress(double force, double area) {
        return force / area;
    }
    
    /**
     * Calculate strain: ε = ΔL/L₀
     * @param deltaLength Change in length in meters
     * @param originalLength Original length in meters
     * @return Strain (dimensionless)
     */
    public static double calculateStrain(double deltaLength, double originalLength) {
        return deltaLength / originalLength;
    }
    
    /**
     * Calculate elastic modulus: E = σ/ε
     * @param stress Stress in Pascals
     * @param strain Strain (dimensionless)
     * @return Elastic modulus in Pascals
     */
    public static double calculateElasticModulus(double stress, double strain) {
        return stress / strain;
    }
    
    /**
     * Calculate beam deflection for simply supported beam with point load at center
     * δ = (F × L³) / (48 × E × I)
     * @param force Applied force in Newtons
     * @param length Beam length in meters
     * @param elasticModulus Elastic modulus in Pascals
     * @param momentOfInertia Second moment of area in m⁴
     * @return Deflection in meters
     */
    public static double calculateBeamDeflection(double force, double length, 
                                               double elasticModulus, double momentOfInertia) {
        return (force * Math.pow(length, 3)) / (48 * elasticModulus * momentOfInertia);
    }
    
    /**
     * Calculate second moment of area for rectangular cross-section
     * I = (b × h³) / 12
     * @param width Width in meters
     * @param height Height in meters
     * @return Second moment of area in m⁴
     */
    public static double rectangularMomentOfInertia(double width, double height) {
        return (width * Math.pow(height, 3)) / 12;
    }
    
    /**
     * Calculate second moment of area for circular cross-section
     * I = π × d⁴ / 64
     * @param diameter Diameter in meters
     * @return Second moment of area in m⁴
     */
    public static double circularMomentOfInertia(double diameter) {
        return Math.PI * Math.pow(diameter, 4) / 64;
    }
    
    /**
     * Calculate torsional shear stress in circular shaft
     * τ = (T × r) / J
     * @param torque Applied torque in N⋅m
     * @param radius Radius at which stress is calculated in meters
     * @param polarMomentOfInertia Polar moment of inertia in m⁴
     * @return Shear stress in Pascals
     */
    public static double calculateTorsionalShearStress(double torque, double radius, 
                                                     double polarMomentOfInertia) {
        return (torque * radius) / polarMomentOfInertia;
    }
    
    /**
     * Calculate polar moment of inertia for solid circular shaft
     * J = π × d⁴ / 32
     * @param diameter Diameter in meters
     * @return Polar moment of inertia in m⁴
     */
    public static double circularPolarMomentOfInertia(double diameter) {
        return Math.PI * Math.pow(diameter, 4) / 32;
    }
    
    /**
     * Calculate angle of twist in circular shaft
     * θ = (T × L) / (G × J)
     * @param torque Applied torque in N⋅m
     * @param length Length of shaft in meters
     * @param shearModulus Shear modulus in Pascals
     * @param polarMomentOfInertia Polar moment of inertia in m⁴
     * @return Angle of twist in radians
     */
    public static double calculateAngleOfTwist(double torque, double length, 
                                             double shearModulus, double polarMomentOfInertia) {
        return (torque * length) / (shearModulus * polarMomentOfInertia);
    }
    
    /**
     * Calculate critical buckling load for column (Euler's formula)
     * P_cr = (π² × E × I) / (K × L)²
     * @param elasticModulus Elastic modulus in Pascals
     * @param momentOfInertia Second moment of area in m⁴
     * @param effectiveLength Effective length in meters
     * @return Critical buckling load in Newtons
     */
    public static double calculateCriticalBucklingLoad(double elasticModulus, 
                                                     double momentOfInertia, double effectiveLength) {
        return (Math.PI * Math.PI * elasticModulus * momentOfInertia) / 
               (effectiveLength * effectiveLength);
    }
    
    /**
     * Calculate thermal stress
     * σ = α × E × ΔT (for constrained thermal expansion)
     * @param thermalExpansionCoeff Coefficient of thermal expansion in /°C
     * @param elasticModulus Elastic modulus in Pascals
     * @param temperatureChange Temperature change in °C
     * @return Thermal stress in Pascals
     */
    public static double calculateThermalStress(double thermalExpansionCoeff, 
                                              double elasticModulus, double temperatureChange) {
        return thermalExpansionCoeff * elasticModulus * temperatureChange;
    }
    
    /**
     * Calculate factor of safety
     * FoS = σ_allowable / σ_actual
     * @param allowableStress Allowable stress in Pascals
     * @param actualStress Actual stress in Pascals
     * @return Factor of safety
     */
    public static double calculateFactorOfSafety(double allowableStress, double actualStress) {
        return allowableStress / actualStress;
    }
    
    public static void main(String[] args) {
        System.out.println("Mechanical Engineering Calculations:");
        System.out.println("====================================");
        
        // Stress and strain calculation
        double force = 10000; // N
        double area = 0.01; // m² (100 cm²)
        double stress = calculateStress(force, area);
        System.out.println("Stress: " + stress/1e6 + " MPa");
        
        double deltaLength = 0.002; // m (2 mm)
        double originalLength = 1.0; // m
        double strain = calculateStrain(deltaLength, originalLength);
        System.out.println("Strain: " + strain);
        
        double elasticModulus = calculateElasticModulus(stress, strain);
        System.out.println("Elastic Modulus: " + elasticModulus/1e9 + " GPa");
        
        // Beam deflection calculation
        double beamLength = 2.0; // m
        double width = 0.1; // m
        double height = 0.2; // m
        double momentOfInertia = rectangularMomentOfInertia(width, height);
        double deflection = calculateBeamDeflection(force, beamLength, STEEL_YOUNGS_MODULUS, momentOfInertia);
        
        System.out.println("\nBeam Analysis:");
        System.out.println("Moment of Inertia: " + momentOfInertia + " m⁴");
        System.out.println("Deflection: " + deflection*1000 + " mm");
        
        // Torsion calculation
        double diameter = 0.05; // m (50 mm)
        double torque = 1000; // N⋅m
        double polarMomentOfInertia = circularPolarMomentOfInertia(diameter);
        double radius = diameter / 2;
        double shearStress = calculateTorsionalShearStress(torque, radius, polarMomentOfInertia);
        
        System.out.println("\nTorsion Analysis:");
        System.out.println("Polar Moment of Inertia: " + polarMomentOfInertia + " m⁴");
        System.out.println("Maximum Shear Stress: " + shearStress/1e6 + " MPa");
        
        // Buckling analysis
        double columnLength = 3.0; // m
        double bucklingLoad = calculateCriticalBucklingLoad(STEEL_YOUNGS_MODULUS, momentOfInertia, columnLength);
        System.out.println("\nBuckling Analysis:");
        System.out.println("Critical Buckling Load: " + bucklingLoad/1000 + " kN");
        
        // Thermal stress
        double thermalCoeff = 12e-6; // /°C for steel
        double tempChange = 50; // °C
        double thermalStress = calculateThermalStress(thermalCoeff, STEEL_YOUNGS_MODULUS, tempChange);
        
        System.out.println("\nThermal Analysis:");
        System.out.println("Thermal Stress: " + thermalStress/1e6 + " MPa");
        
        // Factor of safety
        double yieldStrength = 250e6; // Pa (250 MPa for mild steel)
        double factorOfSafety = calculateFactorOfSafety(yieldStrength, stress);
        System.out.println("\nSafety Analysis:");
        System.out.println("Factor of Safety: " + factorOfSafety);
    }
}
