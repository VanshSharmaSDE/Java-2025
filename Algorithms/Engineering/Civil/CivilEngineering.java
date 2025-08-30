package Algorithms.Engineering.Civil;

/**
 * Civil Engineering Calculations and Algorithms
 * Comprehensive collection of civil engineering formulas and computations
 */
public class CivilEngineering {
    
    /**
     * Structural Engineering Calculations
     */
    public static class StructuralEngineering {
        
        /**
         * Calculate beam deflection for simply supported beam with point load
         * δ = (P * L³) / (48 * E * I)
         * 
         * @param load Point load (N)
         * @param length Beam length (m)
         * @param elasticModulus Elastic modulus (Pa)
         * @param momentOfInertia Second moment of area (m⁴)
         * @return Maximum deflection (m)
         */
        public static double beamDeflectionPointLoad(double load, double length, 
                                                   double elasticModulus, double momentOfInertia) {
            return (load * Math.pow(length, 3)) / (48 * elasticModulus * momentOfInertia);
        }
        
        /**
         * Calculate beam deflection for uniformly distributed load
         * δ = (5 * w * L⁴) / (384 * E * I)
         * 
         * @param distributedLoad Distributed load (N/m)
         * @param length Beam length (m)
         * @param elasticModulus Elastic modulus (Pa)
         * @param momentOfInertia Second moment of area (m⁴)
         * @return Maximum deflection (m)
         */
        public static double beamDeflectionDistributedLoad(double distributedLoad, double length,
                                                         double elasticModulus, double momentOfInertia) {
            return (5 * distributedLoad * Math.pow(length, 4)) / (384 * elasticModulus * momentOfInertia);
        }
        
        /**
         * Calculate maximum bending moment for simply supported beam
         * M = (w * L²) / 8 for distributed load
         * M = (P * L) / 4 for point load at center
         */
        public static double maxBendingMomentDistributed(double distributedLoad, double length) {
            return (distributedLoad * Math.pow(length, 2)) / 8;
        }
        
        public static double maxBendingMomentPointLoad(double pointLoad, double length) {
            return (pointLoad * length) / 4;
        }
        
        /**
         * Calculate critical buckling load for column (Euler's formula)
         * P_cr = (π² * E * I) / (K * L)²
         * 
         * @param elasticModulus Elastic modulus (Pa)
         * @param momentOfInertia Second moment of area (m⁴)
         * @param effectiveLength Effective length (m)
         * @return Critical buckling load (N)
         */
        public static double eulerBucklingLoad(double elasticModulus, double momentOfInertia, 
                                             double effectiveLength) {
            return (Math.PI * Math.PI * elasticModulus * momentOfInertia) / 
                   Math.pow(effectiveLength, 2);
        }
        
        /**
         * Calculate section modulus for rectangular beam
         * S = (b * h²) / 6
         */
        public static double sectionModulusRectangular(double width, double height) {
            return (width * Math.pow(height, 2)) / 6;
        }
        
        /**
         * Calculate section modulus for circular beam
         * S = (π * d³) / 32
         */
        public static double sectionModulusCircular(double diameter) {
            return (Math.PI * Math.pow(diameter, 3)) / 32;
        }
        
        /**
         * Calculate shear stress in beam
         * τ = (V * Q) / (I * t)
         */
        public static double shearStress(double shearForce, double firstMoment, 
                                       double momentOfInertia, double thickness) {
            return (shearForce * firstMoment) / (momentOfInertia * thickness);
        }
    }
    
    /**
     * Geotechnical Engineering Calculations
     */
    public static class GeotechnicalEngineering {
        
        /**
         * Calculate bearing capacity using Terzaghi's formula
         * q_ult = c*N_c + γ*D*N_q + 0.5*γ*B*N_γ
         */
        public static double bearingCapacity(double cohesion, double unitWeight, double depth,
                                           double width, double Nc, double Nq, double Ng) {
            return cohesion * Nc + unitWeight * depth * Nq + 0.5 * unitWeight * width * Ng;
        }
        
        /**
         * Calculate settlement of foundation
         * S = (q * B * (1 - ν²)) / (E_s)
         */
        public static double foundationSettlement(double pressure, double width, 
                                                double poissonRatio, double soilModulus) {
            return (pressure * width * (1 - Math.pow(poissonRatio, 2))) / soilModulus;
        }
        
        /**
         * Calculate active earth pressure (Rankine theory)
         * P_a = 0.5 * γ * H² * K_a
         * K_a = tan²(45° - φ/2)
         */
        public static double activeEarthPressure(double unitWeight, double height, 
                                               double frictionAngle) {
            double Ka = Math.pow(Math.tan(Math.toRadians(45 - frictionAngle / 2)), 2);
            return 0.5 * unitWeight * Math.pow(height, 2) * Ka;
        }
        
        /**
         * Calculate passive earth pressure
         * K_p = tan²(45° + φ/2)
         */
        public static double passiveEarthPressure(double unitWeight, double height, 
                                                double frictionAngle) {
            double Kp = Math.pow(Math.tan(Math.toRadians(45 + frictionAngle / 2)), 2);
            return 0.5 * unitWeight * Math.pow(height, 2) * Kp;
        }
        
        /**
         * Calculate factor of safety for slope stability
         * FS = (c + σ * tan(φ)) / τ
         */
        public static double slopeStabilityFactor(double cohesion, double normalStress,
                                                double frictionAngle, double shearStress) {
            return (cohesion + normalStress * Math.tan(Math.toRadians(frictionAngle))) / shearStress;
        }
        
        /**
         * Calculate consolidation settlement
         * S = (C_c * H * log((σ'₀ + Δσ') / σ'₀)) / (1 + e₀)
         */
        public static double consolidationSettlement(double compressionIndex, double thickness,
                                                   double initialStress, double stressIncrease,
                                                   double initialVoidRatio) {
            return (compressionIndex * thickness * 
                   Math.log10((initialStress + stressIncrease) / initialStress)) / 
                   (1 + initialVoidRatio);
        }
    }
    
    /**
     * Transportation Engineering Calculations
     */
    public static class TransportationEngineering {
        
        /**
         * Calculate stopping sight distance
         * SSD = V²/(2gf) + V*t
         * where V = speed, g = gravity, f = friction, t = reaction time
         */
        public static double stoppingSightDistance(double speed, double friction, double reactionTime) {
            double gravity = 9.81; // m/s²
            return Math.pow(speed, 2) / (2 * gravity * friction) + speed * reactionTime;
        }
        
        /**
         * Calculate superelevation for horizontal curves
         * e = (V²) / (127 * R) - f
         */
        public static double superelevation(double designSpeed, double radius, double friction) {
            return Math.pow(designSpeed, 2) / (127 * radius) - friction;
        }
        
        /**
         * Calculate traffic flow using fundamental equation
         * Flow = Speed × Density
         */
        public static double trafficFlow(double speed, double density) {
            return speed * density;
        }
        
        /**
         * Calculate level of service delay (HCM method)
         * Delay = 0.5 * C * (1 - g/C)² / (1 - X)
         */
        public static double intersectionDelay(double cycleLength, double greenTime, 
                                             double arrivalRate, double capacity) {
            double X = arrivalRate / capacity;
            double gOverC = greenTime / cycleLength;
            return 0.5 * cycleLength * Math.pow(1 - gOverC, 2) / (1 - X);
        }
        
        /**
         * Calculate pavement design thickness (AASHTO method simplified)
         */
        public static double pavementThickness(double designESAL, double soilCBR, 
                                             double reliabilityFactor) {
            // Simplified formula for demonstration
            return Math.log10(designESAL) * reliabilityFactor / Math.sqrt(soilCBR);
        }
    }
    
    /**
     * Hydraulic Engineering Calculations
     */
    public static class HydraulicEngineering {
        
        /**
         * Calculate flow velocity using Manning's equation
         * V = (1/n) * R^(2/3) * S^(1/2)
         */
        public static double manningVelocity(double roughness, double hydraulicRadius, double slope) {
            return (1 / roughness) * Math.pow(hydraulicRadius, 2.0/3) * Math.pow(slope, 0.5);
        }
        
        /**
         * Calculate discharge using continuity equation
         * Q = A * V
         */
        public static double discharge(double area, double velocity) {
            return area * velocity;
        }
        
        /**
         * Calculate pipe flow using Hazen-Williams equation
         * V = 1.318 * C * R^0.63 * S^0.54
         */
        public static double hazenWilliamsVelocity(double coefficient, double hydraulicRadius, 
                                                  double slope) {
            return 1.318 * coefficient * Math.pow(hydraulicRadius, 0.63) * Math.pow(slope, 0.54);
        }
        
        /**
         * Calculate head loss due to friction (Darcy-Weisbach)
         * h_f = f * (L/D) * (V²/2g)
         */
        public static double darcyWeisbachHeadLoss(double frictionFactor, double length, 
                                                  double diameter, double velocity) {
            double gravity = 9.81;
            return frictionFactor * (length / diameter) * (Math.pow(velocity, 2) / (2 * gravity));
        }
        
        /**
         * Calculate critical depth for rectangular channel
         * y_c = (q²/g)^(1/3)
         * where q = Q/b (discharge per unit width)
         */
        public static double criticalDepthRectangular(double discharge, double width) {
            double gravity = 9.81;
            double q = discharge / width;
            return Math.pow(Math.pow(q, 2) / gravity, 1.0/3);
        }
        
        /**
         * Calculate Froude number
         * Fr = V / √(g * D)
         */
        public static double froudeNumber(double velocity, double hydraulicDepth) {
            double gravity = 9.81;
            return velocity / Math.sqrt(gravity * hydraulicDepth);
        }
        
        /**
         * Calculate weir discharge (sharp-crested)
         * Q = C * L * H^(3/2)
         */
        public static double weirDischarge(double coefficient, double length, double head) {
            return coefficient * length * Math.pow(head, 1.5);
        }
    }
    
    /**
     * Environmental Engineering Calculations
     */
    public static class EnvironmentalEngineering {
        
        /**
         * Calculate BOD removal in treatment plant
         * BOD_out = BOD_in * e^(-k*t)
         */
        public static double bodRemoval(double influenBOD, double reactionRate, double time) {
            return influenBOD * Math.exp(-reactionRate * time);
        }
        
        /**
         * Calculate chlorine demand
         * Chlorine Demand = Chlorine Applied - Chlorine Residual
         */
        public static double chlorineDemand(double applied, double residual) {
            return applied - residual;
        }
        
        /**
         * Calculate population using arithmetic growth
         * P_n = P_0 + n * K_a
         */
        public static double arithmeticGrowth(double initialPop, int years, double growthRate) {
            return initialPop + years * growthRate;
        }
        
        /**
         * Calculate population using geometric growth
         * P_n = P_0 * (1 + r)^n
         */
        public static double geometricGrowth(double initialPop, int years, double growthRate) {
            return initialPop * Math.pow(1 + growthRate, years);
        }
        
        /**
         * Calculate water demand
         * Total Demand = Per Capita Demand * Population * Peak Factor
         */
        public static double waterDemand(double perCapitaDemand, double population, 
                                       double peakFactor) {
            return perCapitaDemand * population * peakFactor;
        }
    }
    
    /**
     * Construction Engineering Calculations
     */
    public static class ConstructionEngineering {
        
        /**
         * Calculate concrete mix proportions (by weight)
         * Basic 1:2:4 mix ratio calculation
         */
        public static class ConcreteMix {
            public double cement;
            public double sand;
            public double aggregate;
            public double water;
            
            public ConcreteMix(double totalVolume, double cementRatio, double sandRatio, 
                             double aggregateRatio, double waterCementRatio) {
                double totalRatio = cementRatio + sandRatio + aggregateRatio;
                this.cement = (totalVolume * cementRatio) / totalRatio;
                this.sand = (totalVolume * sandRatio) / totalRatio;
                this.aggregate = (totalVolume * aggregateRatio) / totalRatio;
                this.water = this.cement * waterCementRatio;
            }
        }
        
        /**
         * Calculate earthwork volume (prismoidal formula)
         * V = (L/6) * (A1 + 4*Am + A2)
         */
        public static double earthworkVolume(double length, double area1, double areaMid, 
                                           double area2) {
            return (length / 6) * (area1 + 4 * areaMid + area2);
        }
        
        /**
         * Calculate project duration using PERT
         * Expected Time = (Optimistic + 4*Most Likely + Pessimistic) / 6
         */
        public static double pertExpectedTime(double optimistic, double mostLikely, 
                                            double pessimistic) {
            return (optimistic + 4 * mostLikely + pessimistic) / 6;
        }
        
        /**
         * Calculate productivity
         * Productivity = Output / Input
         */
        public static double productivity(double output, double input) {
            return output / input;
        }
        
        /**
         * Calculate cost escalation
         * Escalated Cost = Base Cost * (1 + escalation rate)^years
         */
        public static double costEscalation(double baseCost, double escalationRate, int years) {
            return baseCost * Math.pow(1 + escalationRate, years);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Civil Engineering Calculations Demo:");
        System.out.println("====================================");
        
        // Structural Engineering
        System.out.println("1. Structural Engineering:");
        double deflection = StructuralEngineering.beamDeflectionPointLoad(
            10000, 5.0, 200e9, 8.33e-6);
        System.out.printf("Beam deflection: %.6f m%n", deflection);
        
        double bucklingLoad = StructuralEngineering.eulerBucklingLoad(
            200e9, 8.33e-6, 3.0);
        System.out.printf("Euler buckling load: %.0f N%n", bucklingLoad);
        
        // Geotechnical Engineering
        System.out.println("\n2. Geotechnical Engineering:");
        double bearingCap = GeotechnicalEngineering.bearingCapacity(
            20000, 18000, 1.5, 2.0, 17.7, 7.4, 15.1);
        System.out.printf("Bearing capacity: %.0f Pa%n", bearingCap);
        
        double activePress = GeotechnicalEngineering.activeEarthPressure(
            18000, 5.0, 30.0);
        System.out.printf("Active earth pressure: %.0f Pa%n", activePress);
        
        // Transportation Engineering
        System.out.println("\n3. Transportation Engineering:");
        double ssd = TransportationEngineering.stoppingSightDistance(
            25.0, 0.35, 2.5);
        System.out.printf("Stopping sight distance: %.1f m%n", ssd);
        
        double superelevation = TransportationEngineering.superelevation(
            80.0, 300.0, 0.15);
        System.out.printf("Superelevation: %.3f%n", superelevation);
        
        // Hydraulic Engineering
        System.out.println("\n4. Hydraulic Engineering:");
        double velocity = HydraulicEngineering.manningVelocity(
            0.025, 1.5, 0.001);
        System.out.printf("Manning velocity: %.2f m/s%n", velocity);
        
        double discharge = HydraulicEngineering.discharge(5.0, velocity);
        System.out.printf("Discharge: %.2f m³/s%n", discharge);
        
        double froude = HydraulicEngineering.froudeNumber(velocity, 2.0);
        System.out.printf("Froude number: %.3f%n", froude);
        
        // Environmental Engineering
        System.out.println("\n5. Environmental Engineering:");
        double bodOut = EnvironmentalEngineering.bodRemoval(300, 0.23, 5);
        System.out.printf("BOD after treatment: %.1f mg/L%n", bodOut);
        
        double population = EnvironmentalEngineering.geometricGrowth(
            100000, 10, 0.025);
        System.out.printf("Population after 10 years: %.0f%n", population);
        
        // Construction Engineering
        System.out.println("\n6. Construction Engineering:");
        ConstructionEngineering.ConcreteMix mix = 
            new ConstructionEngineering.ConcreteMix(1.0, 1, 2, 4, 0.5);
        System.out.printf("Concrete mix for 1 m³:%n");
        System.out.printf("  Cement: %.2f m³%n", mix.cement);
        System.out.printf("  Sand: %.2f m³%n", mix.sand);
        System.out.printf("  Aggregate: %.2f m³%n", mix.aggregate);
        System.out.printf("  Water: %.2f m³%n", mix.water);
        
        double earthwork = ConstructionEngineering.earthworkVolume(
            100, 50, 75, 60);
        System.out.printf("Earthwork volume: %.0f m³%n", earthwork);
        
        double pertTime = ConstructionEngineering.pertExpectedTime(5, 8, 15);
        System.out.printf("PERT expected time: %.1f days%n", pertTime);
        
        double escalatedCost = ConstructionEngineering.costEscalation(
            1000000, 0.03, 5);
        System.out.printf("Escalated cost after 5 years: $%.0f%n", escalatedCost);
    }
}
