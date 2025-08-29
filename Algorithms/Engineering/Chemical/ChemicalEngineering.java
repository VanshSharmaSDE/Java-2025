package Algorithms.Engineering.Chemical;

/**
 * Chemical Engineering Algorithms and Calculations
 */
public class ChemicalEngineering {
    
    // Physical constants
    private static final double GAS_CONSTANT = 8.314; // J/(mol·K)
    private static final double AVOGADRO_NUMBER = 6.022e23; // mol⁻¹
    private static final double BOLTZMANN_CONSTANT = 1.381e-23; // J/K
    
    /**
     * Mass Transfer and Reaction Engineering
     */
    public static class MassTransfer {
        
        /**
         * Calculate Reynolds number: Re = ρvD/μ
         * @param density Fluid density (kg/m³)
         * @param velocity Fluid velocity (m/s)
         * @param diameter Characteristic diameter (m)
         * @param viscosity Dynamic viscosity (Pa·s)
         * @return Reynolds number
         */
        public static double reynoldsNumber(double density, double velocity, 
                                          double diameter, double viscosity) {
            return (density * velocity * diameter) / viscosity;
        }
        
        /**
         * Calculate Schmidt number: Sc = μ/(ρD)
         * @param viscosity Dynamic viscosity (Pa·s)
         * @param density Fluid density (kg/m³)
         * @param diffusivity Mass diffusivity (m²/s)
         * @return Schmidt number
         */
        public static double schmidtNumber(double viscosity, double density, double diffusivity) {
            return viscosity / (density * diffusivity);
        }
        
        /**
         * Calculate Sherwood number correlation for mass transfer
         * Sh = 0.023 * Re^0.8 * Sc^0.33 (for turbulent flow in pipes)
         * @param reynoldsNumber Reynolds number
         * @param schmidtNumber Schmidt number
         * @return Sherwood number
         */
        public static double sherwoodNumber(double reynoldsNumber, double schmidtNumber) {
            return 0.023 * Math.pow(reynoldsNumber, 0.8) * Math.pow(schmidtNumber, 0.33);
        }
        
        /**
         * Calculate mass transfer coefficient
         * kc = (Sh * D) / L
         * @param sherwoodNumber Sherwood number
         * @param diffusivity Mass diffusivity (m²/s)
         * @param characteristicLength Characteristic length (m)
         * @return Mass transfer coefficient (m/s)
         */
        public static double massTransferCoefficient(double sherwoodNumber, 
                                                   double diffusivity, double characteristicLength) {
            return (sherwoodNumber * diffusivity) / characteristicLength;
        }
        
        /**
         * Calculate mass transfer rate: N = kc * A * ΔC
         * @param massTransferCoeff Mass transfer coefficient (m/s)
         * @param area Transfer area (m²)
         * @param concentrationDiff Concentration difference (mol/m³)
         * @return Mass transfer rate (mol/s)
         */
        public static double massTransferRate(double massTransferCoeff, 
                                            double area, double concentrationDiff) {
            return massTransferCoeff * area * concentrationDiff;
        }
    }
    
    /**
     * Heat Transfer Calculations
     */
    public static class HeatTransfer {
        
        /**
         * Calculate Prandtl number: Pr = Cp*μ/k
         * @param specificHeat Specific heat capacity (J/(kg·K))
         * @param viscosity Dynamic viscosity (Pa·s)
         * @param thermalConductivity Thermal conductivity (W/(m·K))
         * @return Prandtl number
         */
        public static double prandtlNumber(double specificHeat, double viscosity, 
                                         double thermalConductivity) {
            return (specificHeat * viscosity) / thermalConductivity;
        }
        
        /**
         * Calculate Nusselt number correlation for heat transfer
         * Nu = 0.023 * Re^0.8 * Pr^0.4 (for turbulent flow in pipes, heating)
         * @param reynoldsNumber Reynolds number
         * @param prandtlNumber Prandtl number
         * @return Nusselt number
         */
        public static double nusseltNumber(double reynoldsNumber, double prandtlNumber) {
            return 0.023 * Math.pow(reynoldsNumber, 0.8) * Math.pow(prandtlNumber, 0.4);
        }
        
        /**
         * Calculate heat transfer coefficient
         * h = (Nu * k) / D
         * @param nusseltNumber Nusselt number
         * @param thermalConductivity Thermal conductivity (W/(m·K))
         * @param diameter Characteristic diameter (m)
         * @return Heat transfer coefficient (W/(m²·K))
         */
        public static double heatTransferCoefficient(double nusseltNumber, 
                                                   double thermalConductivity, double diameter) {
            return (nusseltNumber * thermalConductivity) / diameter;
        }
        
        /**
         * Calculate heat transfer rate: Q = h * A * ΔT
         * @param heatTransferCoeff Heat transfer coefficient (W/(m²·K))
         * @param area Heat transfer area (m²)
         * @param temperatureDiff Temperature difference (K)
         * @return Heat transfer rate (W)
         */
        public static double heatTransferRate(double heatTransferCoeff, 
                                            double area, double temperatureDiff) {
            return heatTransferCoeff * area * temperatureDiff;
        }
        
        /**
         * Calculate overall heat transfer coefficient for composite wall
         * 1/U = 1/h1 + Σ(L/k) + 1/h2
         * @param h1 Inside heat transfer coefficient (W/(m²·K))
         * @param h2 Outside heat transfer coefficient (W/(m²·K))
         * @param wallThickness Wall thickness (m)
         * @param wallConductivity Wall thermal conductivity (W/(m·K))
         * @return Overall heat transfer coefficient (W/(m²·K))
         */
        public static double overallHeatTransferCoefficient(double h1, double h2, 
                                                          double wallThickness, double wallConductivity) {
            double thermalResistance = (1.0 / h1) + (wallThickness / wallConductivity) + (1.0 / h2);
            return 1.0 / thermalResistance;
        }
    }
    
    /**
     * Reaction Engineering Calculations
     */
    public static class ReactionEngineering {
        
        /**
         * Calculate reaction rate using Arrhenius equation
         * k = A * exp(-Ea/(RT))
         * @param preExponentialFactor Pre-exponential factor (1/s or appropriate units)
         * @param activationEnergy Activation energy (J/mol)
         * @param temperature Temperature (K)
         * @return Rate constant
         */
        public static double arrheniusRateConstant(double preExponentialFactor, 
                                                 double activationEnergy, double temperature) {
            return preExponentialFactor * Math.exp(-activationEnergy / (GAS_CONSTANT * temperature));
        }
        
        /**
         * Calculate conversion for first-order reaction in batch reactor
         * X = 1 - exp(-kt)
         * @param rateConstant Rate constant (1/s)
         * @param time Reaction time (s)
         * @return Conversion fraction
         */
        public static double firstOrderConversion(double rateConstant, double time) {
            return 1.0 - Math.exp(-rateConstant * time);
        }
        
        /**
         * Calculate space time for CSTR (Continuous Stirred Tank Reactor)
         * τ = V/Q = CA0*X/(-rA)
         * @param initialConcentration Initial concentration (mol/m³)
         * @param conversion Conversion fraction
         * @param reactionRate Reaction rate (mol/(m³·s))
         * @return Space time (s)
         */
        public static double cstrSpaceTime(double initialConcentration, 
                                         double conversion, double reactionRate) {
            return (initialConcentration * conversion) / reactionRate;
        }
        
        /**
         * Calculate reactor volume for CSTR
         * V = Q * τ
         * @param volumetricFlowRate Volumetric flow rate (m³/s)
         * @param spaceTime Space time (s)
         * @return Reactor volume (m³)
         */
        public static double cstrVolume(double volumetricFlowRate, double spaceTime) {
            return volumetricFlowRate * spaceTime;
        }
        
        /**
         * Calculate equilibrium constant from thermodynamics
         * K = exp(-ΔG°/(RT))
         * @param standardGibbsEnergy Standard Gibbs energy change (J/mol)
         * @param temperature Temperature (K)
         * @return Equilibrium constant
         */
        public static double equilibriumConstant(double standardGibbsEnergy, double temperature) {
            return Math.exp(-standardGibbsEnergy / (GAS_CONSTANT * temperature));
        }
    }
    
    /**
     * Separation Processes
     */
    public static class SeparationProcesses {
        
        /**
         * Calculate relative volatility for distillation
         * α = (yA/xA) / (yB/xB)
         * @param yA Vapor mole fraction of component A
         * @param xA Liquid mole fraction of component A
         * @param yB Vapor mole fraction of component B
         * @param xB Liquid mole fraction of component B
         * @return Relative volatility
         */
        public static double relativeVolatility(double yA, double xA, double yB, double xB) {
            return (yA / xA) / (yB / xB);
        }
        
        /**
         * Calculate minimum number of theoretical plates using Fenske equation
         * Nmin = ln[(xD,LK/xD,HK) * (xB,HK/xB,LK)] / ln(α)
         * @param xD_LK Distillate mole fraction of light key
         * @param xD_HK Distillate mole fraction of heavy key
         * @param xB_HK Bottom mole fraction of heavy key
         * @param xB_LK Bottom mole fraction of light key
         * @param relativeVolatility Average relative volatility
         * @return Minimum number of theoretical plates
         */
        public static double fenskeEquation(double xD_LK, double xD_HK, 
                                          double xB_HK, double xB_LK, double relativeVolatility) {
            double numerator = Math.log((xD_LK / xD_HK) * (xB_HK / xB_LK));
            double denominator = Math.log(relativeVolatility);
            return numerator / denominator;
        }
        
        /**
         * Calculate theoretical plates using McCabe-Thiele method approximation
         * @param relativeVolatility Relative volatility
         * @param refluxRatio Reflux ratio (L/D)
         * @param feedQuality Feed quality parameter
         * @param topComposition Top product composition
         * @param bottomComposition Bottom product composition
         * @return Approximate number of theoretical plates
         */
        public static double mccabeThieleApprox(double relativeVolatility, double refluxRatio, 
                                              double feedQuality, double topComposition, double bottomComposition) {
            // Simplified approximation
            double minRefluxRatio = (topComposition / (1 - topComposition)) * 
                                  ((1 - bottomComposition) / bottomComposition) / relativeVolatility;
            
            double ratio = refluxRatio / minRefluxRatio;
            return Math.log(topComposition / bottomComposition) / Math.log(relativeVolatility * ratio);
        }
    }
    
    /**
     * Fluid Mechanics in Chemical Engineering
     */
    public static class FluidMechanics {
        
        /**
         * Calculate pressure drop in pipe using Darcy-Weisbach equation
         * ΔP = f * (L/D) * (ρv²/2)
         * @param frictionFactor Darcy friction factor
         * @param length Pipe length (m)
         * @param diameter Pipe diameter (m)
         * @param density Fluid density (kg/m³)
         * @param velocity Fluid velocity (m/s)
         * @return Pressure drop (Pa)
         */
        public static double darcyWeisbachPressureDrop(double frictionFactor, double length, 
                                                     double diameter, double density, double velocity) {
            return frictionFactor * (length / diameter) * (density * velocity * velocity / 2.0);
        }
        
        /**
         * Calculate friction factor for laminar flow
         * f = 64/Re (for Re < 2300)
         * @param reynoldsNumber Reynolds number
         * @return Friction factor
         */
        public static double laminarFrictionFactor(double reynoldsNumber) {
            return 64.0 / reynoldsNumber;
        }
        
        /**
         * Calculate friction factor for turbulent flow using Blasius equation
         * f = 0.316/Re^0.25 (for 3000 < Re < 100000)
         * @param reynoldsNumber Reynolds number
         * @return Friction factor
         */
        public static double turbulentFrictionFactor(double reynoldsNumber) {
            return 0.316 / Math.pow(reynoldsNumber, 0.25);
        }
        
        /**
         * Calculate pump head using Bernoulli's equation
         * H = (P2-P1)/ρg + (v2²-v1²)/(2g) + (z2-z1) + hf
         * @param pressureDiff Pressure difference (Pa)
         * @param density Fluid density (kg/m³)
         * @param velocityDiff Velocity difference squared (m²/s²)
         * @param heightDiff Height difference (m)
         * @param frictionLoss Friction loss (m)
         * @return Pump head (m)
         */
        public static double pumpHead(double pressureDiff, double density, 
                                    double velocityDiff, double heightDiff, double frictionLoss) {
            double g = 9.81; // Acceleration due to gravity
            return (pressureDiff / (density * g)) + (velocityDiff / (2 * g)) + heightDiff + frictionLoss;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Chemical Engineering Calculations:");
        System.out.println("==================================");
        
        // Mass Transfer Example
        System.out.println("Mass Transfer Analysis:");
        double density = 1000; // kg/m³ (water)
        double velocity = 2.0; // m/s
        double diameter = 0.1; // m
        double viscosity = 1e-3; // Pa·s
        double diffusivity = 1e-9; // m²/s
        
        double Re = MassTransfer.reynoldsNumber(density, velocity, diameter, viscosity);
        double Sc = MassTransfer.schmidtNumber(viscosity, density, diffusivity);
        double Sh = MassTransfer.sherwoodNumber(Re, Sc);
        double kc = MassTransfer.massTransferCoefficient(Sh, diffusivity, diameter);
        
        System.out.println("Reynolds number: " + Re);
        System.out.println("Schmidt number: " + Sc);
        System.out.println("Sherwood number: " + Sh);
        System.out.println("Mass transfer coefficient: " + kc + " m/s");
        
        // Heat Transfer Example
        System.out.println("\nHeat Transfer Analysis:");
        double specificHeat = 4186; // J/(kg·K) for water
        double thermalConductivity = 0.6; // W/(m·K) for water
        
        double Pr = HeatTransfer.prandtlNumber(specificHeat, viscosity, thermalConductivity);
        double Nu = HeatTransfer.nusseltNumber(Re, Pr);
        double h = HeatTransfer.heatTransferCoefficient(Nu, thermalConductivity, diameter);
        
        System.out.println("Prandtl number: " + Pr);
        System.out.println("Nusselt number: " + Nu);
        System.out.println("Heat transfer coefficient: " + h + " W/(m²·K)");
        
        // Reaction Engineering Example
        System.out.println("\nReaction Engineering:");
        double A = 1e10; // Pre-exponential factor (1/s)
        double Ea = 50000; // Activation energy (J/mol)
        double T = 350; // Temperature (K)
        double time = 3600; // 1 hour
        
        double k = ReactionEngineering.arrheniusRateConstant(A, Ea, T);
        double conversion = ReactionEngineering.firstOrderConversion(k, time);
        
        System.out.println("Rate constant: " + k + " 1/s");
        System.out.println("Conversion after 1 hour: " + (conversion * 100) + "%");
        
        // Separation Process Example
        System.out.println("\nDistillation Column Design:");
        double alpha = 2.5; // Relative volatility
        double xD_LK = 0.95, xD_HK = 0.05;
        double xB_LK = 0.05, xB_HK = 0.95;
        
        double Nmin = SeparationProcesses.fenskeEquation(xD_LK, xD_HK, xB_HK, xB_LK, alpha);
        System.out.println("Minimum theoretical plates: " + Nmin);
        
        // Fluid Mechanics Example
        System.out.println("\nFluid Mechanics:");
        double length = 100; // m
        double frictionFactor = FluidMechanics.laminarFrictionFactor(Re);
        if (Re > 3000) {
            frictionFactor = FluidMechanics.turbulentFrictionFactor(Re);
        }
        
        double pressureDrop = FluidMechanics.darcyWeisbachPressureDrop(
            frictionFactor, length, diameter, density, velocity);
        
        System.out.println("Friction factor: " + frictionFactor);
        System.out.println("Pressure drop: " + pressureDrop + " Pa");
    }
}
