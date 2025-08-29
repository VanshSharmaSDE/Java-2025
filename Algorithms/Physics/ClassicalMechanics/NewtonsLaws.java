package Algorithms.Physics.ClassicalMechanics;

/**
 * Newton's Laws of Motion and Classical Mechanics Algorithms
 */
public class NewtonsLaws {
    
    /**
     * Newton's First Law: F = ma (when mass is constant)
     * Calculate force given mass and acceleration
     * @param mass Mass in kg
     * @param acceleration Acceleration in m/s²
     * @return Force in Newtons
     */
    public static double calculateForce(double mass, double acceleration) {
        return mass * acceleration;
    }
    
    /**
     * Calculate acceleration given force and mass
     * @param force Force in Newtons
     * @param mass Mass in kg
     * @return Acceleration in m/s²
     */
    public static double calculateAcceleration(double force, double mass) {
        return force / mass;
    }
    
    /**
     * Calculate kinetic energy: KE = ½mv²
     * @param mass Mass in kg
     * @param velocity Velocity in m/s
     * @return Kinetic energy in Joules
     */
    public static double calculateKineticEnergy(double mass, double velocity) {
        return 0.5 * mass * velocity * velocity;
    }
    
    /**
     * Calculate potential energy: PE = mgh
     * @param mass Mass in kg
     * @param gravity Gravitational acceleration in m/s² (9.81 on Earth)
     * @param height Height in meters
     * @return Potential energy in Joules
     */
    public static double calculatePotentialEnergy(double mass, double gravity, double height) {
        return mass * gravity * height;
    }
    
    /**
     * Calculate velocity using kinematic equation: v = u + at
     * @param initialVelocity Initial velocity in m/s
     * @param acceleration Acceleration in m/s²
     * @param time Time in seconds
     * @return Final velocity in m/s
     */
    public static double calculateFinalVelocity(double initialVelocity, double acceleration, double time) {
        return initialVelocity + acceleration * time;
    }
    
    /**
     * Calculate displacement using kinematic equation: s = ut + ½at²
     * @param initialVelocity Initial velocity in m/s
     * @param acceleration Acceleration in m/s²
     * @param time Time in seconds
     * @return Displacement in meters
     */
    public static double calculateDisplacement(double initialVelocity, double acceleration, double time) {
        return initialVelocity * time + 0.5 * acceleration * time * time;
    }
    
    /**
     * Calculate momentum: p = mv
     * @param mass Mass in kg
     * @param velocity Velocity in m/s
     * @return Momentum in kg⋅m/s
     */
    public static double calculateMomentum(double mass, double velocity) {
        return mass * velocity;
    }
    
    /**
     * Calculate work done: W = F⋅d⋅cos(θ)
     * @param force Force in Newtons
     * @param displacement Displacement in meters
     * @param angle Angle between force and displacement in radians
     * @return Work done in Joules
     */
    public static double calculateWork(double force, double displacement, double angle) {
        return force * displacement * Math.cos(angle);
    }
    
    public static void main(String[] args) {
        // Example calculations
        double mass = 10.0; // kg
        double acceleration = 9.81; // m/s²
        double velocity = 20.0; // m/s
        double height = 100.0; // m
        double time = 5.0; // s
        
        System.out.println("Classical Mechanics Calculations:");
        System.out.println("================================");
        
        double force = calculateForce(mass, acceleration);
        System.out.println("Force (F = ma): " + force + " N");
        
        double kineticEnergy = calculateKineticEnergy(mass, velocity);
        System.out.println("Kinetic Energy (KE = ½mv²): " + kineticEnergy + " J");
        
        double potentialEnergy = calculatePotentialEnergy(mass, 9.81, height);
        System.out.println("Potential Energy (PE = mgh): " + potentialEnergy + " J");
        
        double finalVelocity = calculateFinalVelocity(0, acceleration, time);
        System.out.println("Final Velocity (v = u + at): " + finalVelocity + " m/s");
        
        double displacement = calculateDisplacement(0, acceleration, time);
        System.out.println("Displacement (s = ut + ½at²): " + displacement + " m");
        
        double momentum = calculateMomentum(mass, velocity);
        System.out.println("Momentum (p = mv): " + momentum + " kg⋅m/s");
        
        double work = calculateWork(force, 10.0, 0); // Force parallel to displacement
        System.out.println("Work Done (W = F⋅d): " + work + " J");
    }
}
