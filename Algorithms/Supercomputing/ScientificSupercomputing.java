package Algorithms.Supercomputing;

import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.*;

/**
 * Advanced Parallel Computing Algorithms for Supercomputing
 * Molecular dynamics, weather simulation, and scientific computing
 */
public class ScientificSupercomputing {
    
    /**
     * Molecular Dynamics Simulation
     */
    public static class MolecularDynamics {
        
        public static class Particle {
            public double x, y, z;           // Position
            public double vx, vy, vz;        // Velocity
            public double fx, fy, fz;        // Force
            public double mass;
            public int type;
            
            public Particle(double x, double y, double z, double mass, int type) {
                this.x = x; this.y = y; this.z = z;
                this.mass = mass;
                this.type = type;
                this.vx = this.vy = this.vz = 0;
                this.fx = this.fy = this.fz = 0;
            }
            
            public double distanceTo(Particle other) {
                double dx = x - other.x;
                double dy = y - other.y;
                double dz = z - other.z;
                return Math.sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
        
        public static class MDSimulation {
            private List<Particle> particles;
            private double boxSize;
            private double timeStep;
            private double temperature;
            private static final double BOLTZMANN = 1.380649e-23;
            private static final double EPSILON = 1.0;  // LJ potential parameter
            private static final double SIGMA = 1.0;    // LJ potential parameter
            private ExecutorService executor;
            
            public MDSimulation(int numParticles, double boxSize, double timeStep, double temperature) {
                this.boxSize = boxSize;
                this.timeStep = timeStep;
                this.temperature = temperature;
                this.particles = new ArrayList<>();
                this.executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                
                initializeParticles(numParticles);
            }
            
            private void initializeParticles(int numParticles) {
                Random random = new Random();
                
                for (int i = 0; i < numParticles; i++) {
                    double x = random.nextDouble() * boxSize;
                    double y = random.nextDouble() * boxSize;
                    double z = random.nextDouble() * boxSize;
                    
                    Particle particle = new Particle(x, y, z, 1.0, 0);
                    
                    // Initialize velocities from Maxwell-Boltzmann distribution
                    double sigma = Math.sqrt(BOLTZMANN * temperature / particle.mass);
                    particle.vx = random.nextGaussian() * sigma;
                    particle.vy = random.nextGaussian() * sigma;
                    particle.vz = random.nextGaussian() * sigma;
                    
                    particles.add(particle);
                }
            }
            
            public void simulate(int numSteps) {
                for (int step = 0; step < numSteps; step++) {
                    calculateForces();
                    updatePositionsAndVelocities();
                    applyPeriodicBoundaryConditions();
                    
                    if (step % 100 == 0) {
                        double energy = calculateTotalEnergy();
                        double temp = calculateTemperature();
                        System.out.printf("Step %d: Energy = %.3f, Temperature = %.3f K\n", 
                                         step, energy, temp);
                    }
                }
            }
            
            private void calculateForces() {
                // Reset forces
                particles.parallelStream().forEach(p -> {
                    p.fx = p.fy = p.fz = 0;
                });
                
                // Calculate pairwise forces in parallel
                int numParticles = particles.size();
                List<CompletableFuture<Void>> futures = new ArrayList<>();
                
                int numThreads = Runtime.getRuntime().availableProcessors();
                int chunkSize = (numParticles * (numParticles - 1)) / (2 * numThreads);
                
                AtomicInteger pairIndex = new AtomicInteger(0);
                
                for (int t = 0; t < numThreads; t++) {
                    futures.add(CompletableFuture.runAsync(() -> {
                        while (true) {
                            int idx = pairIndex.getAndIncrement();
                            if (idx >= numParticles * (numParticles - 1) / 2) break;
                            
                            // Convert linear index to (i,j) pair
                            int i = 0, j = 0;
                            int count = 0;
                            outer: for (int ii = 0; ii < numParticles; ii++) {
                                for (int jj = ii + 1; jj < numParticles; jj++) {
                                    if (count == idx) {
                                        i = ii; j = jj;
                                        break outer;
                                    }
                                    count++;
                                }
                            }
                            
                            if (i < numParticles && j < numParticles) {
                                calculatePairwiseForce(particles.get(i), particles.get(j));
                            }
                        }
                    }, executor));
                }
                
                // Wait for all force calculations to complete
                for (CompletableFuture<Void> future : futures) {
                    try {
                        future.get();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            
            private void calculatePairwiseForce(Particle p1, Particle p2) {
                double dx = p1.x - p2.x;
                double dy = p1.y - p2.y;
                double dz = p1.z - p2.z;
                
                // Apply minimum image convention
                dx = dx - boxSize * Math.round(dx / boxSize);
                dy = dy - boxSize * Math.round(dy / boxSize);
                dz = dz - boxSize * Math.round(dz / boxSize);
                
                double r2 = dx*dx + dy*dy + dz*dz;
                double r = Math.sqrt(r2);
                
                if (r < 3.0 * SIGMA) { // Cutoff distance
                    // Lennard-Jones potential force
                    double r6 = Math.pow(SIGMA/r, 6);
                    double r12 = r6 * r6;
                    double force_magnitude = 24 * EPSILON * (2*r12 - r6) / r2;
                    
                    double fx = force_magnitude * dx;
                    double fy = force_magnitude * dy;
                    double fz = force_magnitude * dz;
                    
                    // Newton's third law
                    synchronized(p1) {
                        p1.fx += fx;
                        p1.fy += fy;
                        p1.fz += fz;
                    }
                    
                    synchronized(p2) {
                        p2.fx -= fx;
                        p2.fy -= fy;
                        p2.fz -= fz;
                    }
                }
            }
            
            private void updatePositionsAndVelocities() {
                // Velocity Verlet integration
                particles.parallelStream().forEach(p -> {
                    // Update positions
                    p.x += p.vx * timeStep + 0.5 * (p.fx / p.mass) * timeStep * timeStep;
                    p.y += p.vy * timeStep + 0.5 * (p.fy / p.mass) * timeStep * timeStep;
                    p.z += p.vz * timeStep + 0.5 * (p.fz / p.mass) * timeStep * timeStep;
                    
                    // Update velocities (half step)
                    p.vx += 0.5 * (p.fx / p.mass) * timeStep;
                    p.vy += 0.5 * (p.fy / p.mass) * timeStep;
                    p.vz += 0.5 * (p.fz / p.mass) * timeStep;
                });
            }
            
            private void applyPeriodicBoundaryConditions() {
                particles.parallelStream().forEach(p -> {
                    p.x = ((p.x % boxSize) + boxSize) % boxSize;
                    p.y = ((p.y % boxSize) + boxSize) % boxSize;
                    p.z = ((p.z % boxSize) + boxSize) % boxSize;
                });
            }
            
            private double calculateTotalEnergy() {
                double kineticEnergy = particles.parallelStream()
                    .mapToDouble(p -> 0.5 * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz))
                    .sum();
                
                double potentialEnergy = 0;
                for (int i = 0; i < particles.size(); i++) {
                    for (int j = i + 1; j < particles.size(); j++) {
                        double r = particles.get(i).distanceTo(particles.get(j));
                        if (r < 3.0 * SIGMA) {
                            double r6 = Math.pow(SIGMA/r, 6);
                            potentialEnergy += 4 * EPSILON * (r6*r6 - r6);
                        }
                    }
                }
                
                return kineticEnergy + potentialEnergy;
            }
            
            private double calculateTemperature() {
                double totalKineticEnergy = particles.parallelStream()
                    .mapToDouble(p -> 0.5 * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz))
                    .sum();
                
                return (2.0 / 3.0) * totalKineticEnergy / (particles.size() * BOLTZMANN);
            }
            
            public void shutdown() {
                executor.shutdown();
            }
        }
    }
    
    /**
     * Weather Simulation using Finite Difference Methods
     */
    public static class WeatherSimulation {
        
        public static class AtmosphericGrid {
            private double[][][] temperature;  // [x][y][z]
            private double[][][] pressure;
            private double[][][] humidity;
            private double[][][] windU, windV, windW; // Wind components
            private int nx, ny, nz;
            private double dx, dy, dz, dt;
            private ExecutorService executor;
            
            public AtmosphericGrid(int nx, int ny, int nz, double dx, double dy, double dz, double dt) {
                this.nx = nx; this.ny = ny; this.nz = nz;
                this.dx = dx; this.dy = dy; this.dz = dz; this.dt = dt;
                
                // Initialize arrays
                temperature = new double[nx][ny][nz];
                pressure = new double[nx][ny][nz];
                humidity = new double[nx][ny][nz];
                windU = new double[nx][ny][nz];
                windV = new double[nx][ny][nz];
                windW = new double[nx][ny][nz];
                
                executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                
                initializeAtmosphere();
            }
            
            private void initializeAtmosphere() {
                Random random = new Random();
                
                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < ny; j++) {
                        for (int k = 0; k < nz; k++) {
                            // Temperature decreases with altitude
                            double altitude = k * dz;
                            temperature[i][j][k] = 288.15 - 0.0065 * altitude + random.nextGaussian() * 2;
                            
                            // Pressure decreases exponentially with altitude
                            pressure[i][j][k] = 101325 * Math.exp(-0.00012 * altitude);
                            
                            // Humidity varies randomly
                            humidity[i][j][k] = 0.3 + 0.4 * random.nextDouble();
                            
                            // Initialize small random wind components
                            windU[i][j][k] = random.nextGaussian() * 5;
                            windV[i][j][k] = random.nextGaussian() * 5;
                            windW[i][j][k] = random.nextGaussian() * 1;
                        }
                    }
                }
            }
            
            public void simulate(int numTimeSteps) {
                for (int step = 0; step < numTimeSteps; step++) {
                    updateTemperature();
                    updatePressure();
                    updateHumidity();
                    updateWind();
                    applyBoundaryConditions();
                    
                    if (step % 10 == 0) {
                        double avgTemp = calculateAverageTemperature();
                        double maxWind = calculateMaxWindSpeed();
                        System.out.printf("Time step %d: Avg temp = %.2f K, Max wind = %.2f m/s\n",
                                         step, avgTemp, maxWind);
                    }
                }
            }
            
            private void updateTemperature() {
                double[][][] newTemp = new double[nx][ny][nz];
                List<CompletableFuture<Void>> futures = new ArrayList<>();
                
                int numThreads = Runtime.getRuntime().availableProcessors();
                int slicesPerThread = Math.max(1, nz / numThreads);
                
                for (int t = 0; t < numThreads; t++) {
                    final int startK = t * slicesPerThread;
                    final int endK = Math.min((t + 1) * slicesPerThread, nz);
                    
                    futures.add(CompletableFuture.runAsync(() -> {
                        for (int k = startK; k < endK; k++) {
                            for (int i = 1; i < nx - 1; i++) {
                                for (int j = 1; j < ny - 1; j++) {
                                    // Heat diffusion equation
                                    double alpha = 0.1; // Thermal diffusivity
                                    
                                    double d2Tdx2 = (temperature[i+1][j][k] - 2*temperature[i][j][k] + temperature[i-1][j][k]) / (dx*dx);
                                    double d2Tdy2 = (temperature[i][j+1][k] - 2*temperature[i][j][k] + temperature[i][j-1][k]) / (dy*dy);
                                    double d2Tdz2 = k > 0 && k < nz-1 ? 
                                        (temperature[i][j][k+1] - 2*temperature[i][j][k] + temperature[i][j][k-1]) / (dz*dz) : 0;
                                    
                                    // Advection term
                                    double dTdx = (temperature[i+1][j][k] - temperature[i-1][j][k]) / (2*dx);
                                    double dTdy = (temperature[i][j+1][k] - temperature[i][j-1][k]) / (2*dy);
                                    
                                    double advection = windU[i][j][k] * dTdx + windV[i][j][k] * dTdy;
                                    
                                    newTemp[i][j][k] = temperature[i][j][k] + dt * (alpha * (d2Tdx2 + d2Tdy2 + d2Tdz2) - advection);
                                }
                            }
                        }
                    }, executor));
                }
                
                // Wait for completion
                futures.forEach(f -> {
                    try { f.get(); } catch (Exception e) { e.printStackTrace(); }
                });
                
                temperature = newTemp;
            }
            
            private void updatePressure() {
                // Simplified pressure update based on temperature and continuity
                for (int i = 1; i < nx - 1; i++) {
                    for (int j = 1; j < ny - 1; j++) {
                        for (int k = 1; k < nz - 1; k++) {
                            // Ideal gas law relationship
                            double tempRatio = temperature[i][j][k] / 288.15; // Reference temperature
                            pressure[i][j][k] *= tempRatio;
                            
                            // Add divergence correction
                            double divU = (windU[i+1][j][k] - windU[i-1][j][k]) / (2*dx) +
                                         (windV[i][j+1][k] - windV[i][j-1][k]) / (2*dy) +
                                         (windW[i][j][k+1] - windW[i][j][k-1]) / (2*dz);
                            
                            pressure[i][j][k] -= dt * pressure[i][j][k] * divU;
                        }
                    }
                }
            }
            
            private void updateHumidity() {
                // Humidity transport and phase changes
                for (int i = 1; i < nx - 1; i++) {
                    for (int j = 1; j < ny - 1; j++) {
                        for (int k = 1; k < nz - 1; k++) {
                            // Advection of humidity
                            double dHdx = (humidity[i+1][j][k] - humidity[i-1][j][k]) / (2*dx);
                            double dHdy = (humidity[i][j+1][k] - humidity[i][j-1][k]) / (2*dy);
                            
                            double advection = windU[i][j][k] * dHdx + windV[i][j][k] * dHdy;
                            
                            humidity[i][j][k] -= dt * advection;
                            
                            // Condensation/evaporation
                            double saturationHumidity = calculateSaturationHumidity(temperature[i][j][k]);
                            if (humidity[i][j][k] > saturationHumidity) {
                                // Condensation occurs
                                double condensation = (humidity[i][j][k] - saturationHumidity) * 0.1;
                                humidity[i][j][k] -= condensation;
                                temperature[i][j][k] += condensation * 2257000 / 1005; // Latent heat release
                            }
                            
                            // Ensure bounds
                            humidity[i][j][k] = Math.max(0, Math.min(1, humidity[i][j][k]));
                        }
                    }
                }
            }
            
            private double calculateSaturationHumidity(double temp) {
                // Simplified saturation humidity calculation
                return 0.622 * 611.2 * Math.exp(17.67 * (temp - 273.15) / (temp - 29.65)) / 101325;
            }
            
            private void updateWind() {
                double[][][] newWindU = new double[nx][ny][nz];
                double[][][] newWindV = new double[nx][ny][nz];
                double[][][] newWindW = new double[nx][ny][nz];
                
                // Navier-Stokes equations (simplified)
                for (int i = 1; i < nx - 1; i++) {
                    for (int j = 1; j < ny - 1; j++) {
                        for (int k = 1; k < nz - 1; k++) {
                            // Pressure gradient force
                            double dPdx = (pressure[i+1][j][k] - pressure[i-1][j][k]) / (2*dx);
                            double dPdy = (pressure[i][j+1][k] - pressure[i][j-1][k]) / (2*dy);
                            double dPdz = (pressure[i][j][k+1] - pressure[i][j][k-1]) / (2*dz);
                            
                            double rho = pressure[i][j][k] / (287.05 * temperature[i][j][k]); // Air density
                            
                            // Viscous diffusion (simplified)
                            double nu = 1.5e-5; // Kinematic viscosity
                            
                            double d2Udx2 = (windU[i+1][j][k] - 2*windU[i][j][k] + windU[i-1][j][k]) / (dx*dx);
                            double d2Udy2 = (windU[i][j+1][k] - 2*windU[i][j][k] + windU[i][j-1][k]) / (dy*dy);
                            
                            double d2Vdx2 = (windV[i+1][j][k] - 2*windV[i][j][k] + windV[i-1][j][k]) / (dx*dx);
                            double d2Vdy2 = (windV[i][j+1][k] - 2*windV[i][j][k] + windV[i][j-1][k]) / (dy*dy);
                            
                            // Update wind components
                            newWindU[i][j][k] = windU[i][j][k] + dt * (-dPdx/rho + nu*(d2Udx2 + d2Udy2));
                            newWindV[i][j][k] = windV[i][j][k] + dt * (-dPdy/rho + nu*(d2Vdx2 + d2Vdy2));
                            newWindW[i][j][k] = windW[i][j][k] + dt * (-dPdz/rho - 9.81); // Include gravity
                            
                            // Add Coriolis force (simplified)
                            double f = 1e-4; // Coriolis parameter
                            newWindU[i][j][k] += dt * f * windV[i][j][k];
                            newWindV[i][j][k] -= dt * f * windU[i][j][k];
                        }
                    }
                }
                
                windU = newWindU;
                windV = newWindV;
                windW = newWindW;
            }
            
            private void applyBoundaryConditions() {
                // Ground boundary conditions
                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < ny; j++) {
                        windW[i][j][0] = 0; // No vertical wind at ground
                        temperature[i][j][0] = 288.15; // Fixed ground temperature
                    }
                }
                
                // Side boundaries (periodic)
                for (int j = 0; j < ny; j++) {
                    for (int k = 0; k < nz; k++) {
                        temperature[0][j][k] = temperature[nx-2][j][k];
                        temperature[nx-1][j][k] = temperature[1][j][k];
                        windU[0][j][k] = windU[nx-2][j][k];
                        windU[nx-1][j][k] = windU[1][j][k];
                    }
                }
            }
            
            private double calculateAverageTemperature() {
                return Arrays.stream(temperature)
                    .flatMap(Arrays::stream)
                    .flatMapToDouble(Arrays::stream)
                    .average()
                    .orElse(0.0);
            }
            
            private double calculateMaxWindSpeed() {
                double maxSpeed = 0;
                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < ny; j++) {
                        for (int k = 0; k < nz; k++) {
                            double speed = Math.sqrt(windU[i][j][k]*windU[i][j][k] + 
                                                   windV[i][j][k]*windV[i][j][k] + 
                                                   windW[i][j][k]*windW[i][j][k]);
                            maxSpeed = Math.max(maxSpeed, speed);
                        }
                    }
                }
                return maxSpeed;
            }
            
            public void shutdown() {
                executor.shutdown();
            }
        }
    }
    
    /**
     * Computational Fluid Dynamics using Lattice Boltzmann Method
     */
    public static class LatticeBoltzmannMethod {
        
        public static class LBMSimulation {
            private double[][][] f;  // Distribution functions [x][y][direction]
            private double[][][] fNew;
            private boolean[][] obstacle;
            private int nx, ny;
            private double tau; // Relaxation time
            private double[] weights;
            private int[][] directions;
            
            // D2Q9 model (2D, 9 velocities)
            private static final int NUM_DIRECTIONS = 9;
            
            public LBMSimulation(int nx, int ny, double tau) {
                this.nx = nx;
                this.ny = ny;
                this.tau = tau;
                
                f = new double[nx][ny][NUM_DIRECTIONS];
                fNew = new double[nx][ny][NUM_DIRECTIONS];
                obstacle = new boolean[nx][ny];
                
                // D2Q9 weights
                weights = new double[]{4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
                
                // D2Q9 directions
                directions = new int[][]{{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, 
                                       {1,1}, {-1,1}, {-1,-1}, {1,-1}};
                
                initializeFlow();
            }
            
            private void initializeFlow() {
                double rho0 = 1.0;
                double u0 = 0.1;
                double v0 = 0.0;
                
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        double[] feq = equilibriumDistribution(rho0, u0, v0);
                        System.arraycopy(feq, 0, f[x][y], 0, NUM_DIRECTIONS);
                    }
                }
                
                // Add circular obstacle
                int centerX = nx / 4;
                int centerY = ny / 2;
                int radius = ny / 10;
                
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        int dx = x - centerX;
                        int dy = y - centerY;
                        obstacle[x][y] = (dx*dx + dy*dy) <= radius*radius;
                    }
                }
            }
            
            public void simulate(int numSteps) {
                for (int step = 0; step < numSteps; step++) {
                    collisionStep();
                    streamingStep();
                    boundaryConditions();
                    
                    if (step % 100 == 0) {
                        double[] avgVelocity = calculateAverageVelocity();
                        System.out.printf("Step %d: Avg velocity = (%.4f, %.4f)\n",
                                         step, avgVelocity[0], avgVelocity[1]);
                    }
                }
            }
            
            private void collisionStep() {
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        if (!obstacle[x][y]) {
                            double[] moments = calculateMoments(f[x][y]);
                            double rho = moments[0];
                            double u = moments[1];
                            double v = moments[2];
                            
                            double[] feq = equilibriumDistribution(rho, u, v);
                            
                            // BGK collision
                            for (int i = 0; i < NUM_DIRECTIONS; i++) {
                                fNew[x][y][i] = f[x][y][i] - (f[x][y][i] - feq[i]) / tau;
                            }
                        }
                    }
                }
                
                // Swap arrays
                double[][][] temp = f;
                f = fNew;
                fNew = temp;
            }
            
            private void streamingStep() {
                // Copy current state
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        System.arraycopy(f[x][y], 0, fNew[x][y], 0, NUM_DIRECTIONS);
                    }
                }
                
                // Stream
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        if (!obstacle[x][y]) {
                            for (int i = 1; i < NUM_DIRECTIONS; i++) {
                                int newX = x + directions[i][0];
                                int newY = y + directions[i][1];
                                
                                // Periodic boundary conditions
                                newX = (newX + nx) % nx;
                                newY = (newY + ny) % ny;
                                
                                if (!obstacle[newX][newY]) {
                                    f[newX][newY][i] = fNew[x][y][i];
                                }
                            }
                        }
                    }
                }
            }
            
            private void boundaryConditions() {
                // Inlet boundary (left side)
                for (int y = 0; y < ny; y++) {
                    if (!obstacle[0][y]) {
                        double rho = 1.0;
                        double u = 0.1;
                        double v = 0.0;
                        
                        double[] feq = equilibriumDistribution(rho, u, v);
                        System.arraycopy(feq, 0, f[0][y], 0, NUM_DIRECTIONS);
                    }
                }
                
                // Outlet boundary (right side) - zero gradient
                for (int y = 0; y < ny; y++) {
                    if (!obstacle[nx-1][y]) {
                        System.arraycopy(f[nx-2][y], 0, f[nx-1][y], 0, NUM_DIRECTIONS);
                    }
                }
                
                // Bounce-back for obstacles
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        if (obstacle[x][y]) {
                            // Bounce back - reverse directions
                            double[] temp = new double[NUM_DIRECTIONS];
                            temp[0] = f[x][y][0];
                            temp[1] = f[x][y][3]; // East -> West
                            temp[2] = f[x][y][4]; // North -> South
                            temp[3] = f[x][y][1]; // West -> East
                            temp[4] = f[x][y][2]; // South -> North
                            temp[5] = f[x][y][7]; // NE -> SW
                            temp[6] = f[x][y][8]; // NW -> SE
                            temp[7] = f[x][y][5]; // SW -> NE
                            temp[8] = f[x][y][6]; // SE -> NW
                            
                            System.arraycopy(temp, 0, f[x][y], 0, NUM_DIRECTIONS);
                        }
                    }
                }
            }
            
            private double[] equilibriumDistribution(double rho, double u, double v) {
                double[] feq = new double[NUM_DIRECTIONS];
                
                for (int i = 0; i < NUM_DIRECTIONS; i++) {
                    double ci_u = directions[i][0] * u + directions[i][1] * v;
                    double u_sqr = u*u + v*v;
                    
                    feq[i] = weights[i] * rho * (1 + 3*ci_u + 4.5*ci_u*ci_u - 1.5*u_sqr);
                }
                
                return feq;
            }
            
            private double[] calculateMoments(double[] distribution) {
                double rho = Arrays.stream(distribution).sum();
                
                double u = 0, v = 0;
                for (int i = 0; i < NUM_DIRECTIONS; i++) {
                    u += distribution[i] * directions[i][0];
                    v += distribution[i] * directions[i][1];
                }
                
                u /= rho;
                v /= rho;
                
                return new double[]{rho, u, v};
            }
            
            private double[] calculateAverageVelocity() {
                double totalU = 0, totalV = 0;
                int count = 0;
                
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        if (!obstacle[x][y]) {
                            double[] moments = calculateMoments(f[x][y]);
                            totalU += moments[1];
                            totalV += moments[2];
                            count++;
                        }
                    }
                }
                
                return new double[]{totalU / count, totalV / count};
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Scientific Supercomputing Demo:");
        System.out.println("===============================");
        
        // Molecular Dynamics Simulation
        System.out.println("1. Molecular Dynamics Simulation:");
        MolecularDynamics.MDSimulation mdSim = 
            new MolecularDynamics.MDSimulation(100, 10.0, 0.001, 300.0);
        
        System.out.println("Running MD simulation...");
        mdSim.simulate(500);
        mdSim.shutdown();
        
        // Weather Simulation
        System.out.println("\n2. Weather Simulation:");
        WeatherSimulation.AtmosphericGrid weather = 
            new WeatherSimulation.AtmosphericGrid(20, 20, 10, 1000, 1000, 500, 1.0);
        
        System.out.println("Running weather simulation...");
        weather.simulate(50);
        weather.shutdown();
        
        // Lattice Boltzmann Method
        System.out.println("\n3. Computational Fluid Dynamics (LBM):");
        LatticeBoltzmannMethod.LBMSimulation lbm = 
            new LatticeBoltzmannMethod.LBMSimulation(100, 50, 1.0);
        
        System.out.println("Running CFD simulation...");
        lbm.simulate(1000);
        
        System.out.println("\nScientific supercomputing demonstration completed!");
    }
}
