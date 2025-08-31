package Algorithms.SignalProcessing;

import java.util.*;
import java.util.function.*;

/**
 * Digital Signal Processing and Audio Processing Algorithms
 * Fourier transforms, filters, audio effects, speech processing
 */
public class SignalProcessingAlgorithms {
    
    /**
     * Complex number class for signal processing
     */
    public static class Complex {
        public final double real, imag;
        
        public Complex(double real, double imag) {
            this.real = real;
            this.imag = imag;
        }
        
        public Complex(double real) {
            this(real, 0);
        }
        
        public Complex add(Complex other) {
            return new Complex(real + other.real, imag + other.imag);
        }
        
        public Complex subtract(Complex other) {
            return new Complex(real - other.real, imag - other.imag);
        }
        
        public Complex multiply(Complex other) {
            return new Complex(real * other.real - imag * other.imag,
                             real * other.imag + imag * other.real);
        }
        
        public Complex multiply(double scalar) {
            return new Complex(real * scalar, imag * scalar);
        }
        
        public Complex divide(Complex other) {
            double denominator = other.real * other.real + other.imag * other.imag;
            return new Complex((real * other.real + imag * other.imag) / denominator,
                             (imag * other.real - real * other.imag) / denominator);
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
        
        public static Complex exp(double theta) {
            return new Complex(Math.cos(theta), Math.sin(theta));
        }
        
        public String toString() {
            if (imag >= 0) {
                return String.format("%.3f + %.3fi", real, imag);
            } else {
                return String.format("%.3f - %.3fi", real, -imag);
            }
        }
    }
    
    /**
     * Fast Fourier Transform and related algorithms
     */
    public static class FourierTransform {
        
        public static Complex[] fft(Complex[] signal) {
            int n = signal.length;
            
            // Base case
            if (n <= 1) {
                return signal.clone();
            }
            
            // Ensure power of 2
            if ((n & (n - 1)) != 0) {
                throw new IllegalArgumentException("Signal length must be a power of 2");
            }
            
            // Divide
            Complex[] even = new Complex[n / 2];
            Complex[] odd = new Complex[n / 2];
            
            for (int i = 0; i < n / 2; i++) {
                even[i] = signal[2 * i];
                odd[i] = signal[2 * i + 1];
            }
            
            // Conquer
            Complex[] evenFFT = fft(even);
            Complex[] oddFFT = fft(odd);
            
            // Combine
            Complex[] result = new Complex[n];
            for (int k = 0; k < n / 2; k++) {
                Complex t = Complex.exp(-2 * Math.PI * k / n).multiply(oddFFT[k]);
                result[k] = evenFFT[k].add(t);
                result[k + n / 2] = evenFFT[k].subtract(t);
            }
            
            return result;
        }
        
        public static Complex[] ifft(Complex[] spectrum) {
            int n = spectrum.length;
            
            // Conjugate the complex numbers
            Complex[] conjugated = new Complex[n];
            for (int i = 0; i < n; i++) {
                conjugated[i] = spectrum[i].conjugate();
            }
            
            // Compute forward FFT
            Complex[] result = fft(conjugated);
            
            // Conjugate and normalize
            for (int i = 0; i < n; i++) {
                result[i] = result[i].conjugate().multiply(1.0 / n);
            }
            
            return result;
        }
        
        public static double[] powerSpectrum(Complex[] spectrum) {
            double[] power = new double[spectrum.length];
            for (int i = 0; i < spectrum.length; i++) {
                power[i] = spectrum[i].magnitude() * spectrum[i].magnitude();
            }
            return power;
        }
        
        public static double[] magnitudeSpectrum(Complex[] spectrum) {
            double[] magnitude = new double[spectrum.length];
            for (int i = 0; i < spectrum.length; i++) {
                magnitude[i] = spectrum[i].magnitude();
            }
            return magnitude;
        }
        
        public static double[] phaseSpectrum(Complex[] spectrum) {
            double[] phase = new double[spectrum.length];
            for (int i = 0; i < spectrum.length; i++) {
                phase[i] = spectrum[i].phase();
            }
            return phase;
        }
        
        // Short-Time Fourier Transform
        public static Complex[][] stft(double[] signal, int windowSize, int hopSize) {
            int numFrames = (signal.length - windowSize) / hopSize + 1;
            Complex[][] spectrogram = new Complex[numFrames][];
            
            // Hanning window
            double[] window = hanningWindow(windowSize);
            
            for (int frame = 0; frame < numFrames; frame++) {
                Complex[] frameSignal = new Complex[windowSize];
                
                for (int i = 0; i < windowSize; i++) {
                    int signalIndex = frame * hopSize + i;
                    if (signalIndex < signal.length) {
                        frameSignal[i] = new Complex(signal[signalIndex] * window[i]);
                    } else {
                        frameSignal[i] = new Complex(0);
                    }
                }
                
                spectrogram[frame] = fft(frameSignal);
            }
            
            return spectrogram;
        }
        
        private static double[] hanningWindow(int size) {
            double[] window = new double[size];
            for (int i = 0; i < size; i++) {
                window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            }
            return window;
        }
        
        // Discrete Cosine Transform
        public static double[] dct(double[] signal) {
            int n = signal.length;
            double[] result = new double[n];
            
            for (int k = 0; k < n; k++) {
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += signal[i] * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
                }
                
                double scale = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
                result[k] = scale * sum;
            }
            
            return result;
        }
        
        public static double[] idct(double[] coefficients) {
            int n = coefficients.length;
            double[] result = new double[n];
            
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    double scale = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
                    sum += scale * coefficients[k] * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
                }
                result[i] = sum;
            }
            
            return result;
        }
    }
    
    /**
     * Digital Filters
     */
    public static class DigitalFilters {
        
        public static class FilterCoefficients {
            public final double[] numerator;   // b coefficients
            public final double[] denominator; // a coefficients
            
            public FilterCoefficients(double[] numerator, double[] denominator) {
                this.numerator = numerator.clone();
                this.denominator = denominator.clone();
            }
        }
        
        public static class FilterState {
            private final double[] inputHistory;
            private final double[] outputHistory;
            private int index;
            
            public FilterState(int numTaps) {
                this.inputHistory = new double[numTaps];
                this.outputHistory = new double[numTaps];
                this.index = 0;
            }
        }
        
        // FIR Filter using convolution
        public static double[] firFilter(double[] signal, double[] impulseResponse) {
            int signalLength = signal.length;
            int filterLength = impulseResponse.length;
            double[] output = new double[signalLength + filterLength - 1];
            
            for (int n = 0; n < output.length; n++) {
                double sum = 0;
                for (int k = 0; k < filterLength; k++) {
                    int signalIndex = n - k;
                    if (signalIndex >= 0 && signalIndex < signalLength) {
                        sum += signal[signalIndex] * impulseResponse[k];
                    }
                }
                output[n] = sum;
            }
            
            return Arrays.copyOf(output, signalLength); // Trim to original length
        }
        
        // IIR Filter implementation
        public static double[] iirFilter(double[] signal, FilterCoefficients coeffs) {
            int n = signal.length;
            double[] output = new double[n];
            
            int numB = coeffs.numerator.length;
            int numA = coeffs.denominator.length;
            
            // Normalize by a[0]
            double a0 = coeffs.denominator[0];
            double[] b = new double[numB];
            double[] a = new double[numA];
            
            for (int i = 0; i < numB; i++) {
                b[i] = coeffs.numerator[i] / a0;
            }
            for (int i = 0; i < numA; i++) {
                a[i] = coeffs.denominator[i] / a0;
            }
            
            for (int i = 0; i < n; i++) {
                // Compute output
                double sum = 0;
                
                // Feedforward (FIR part)
                for (int j = 0; j < numB; j++) {
                    if (i - j >= 0) {
                        sum += b[j] * signal[i - j];
                    }
                }
                
                // Feedback (IIR part)
                for (int j = 1; j < numA; j++) {
                    if (i - j >= 0) {
                        sum -= a[j] * output[i - j];
                    }
                }
                
                output[i] = sum;
            }
            
            return output;
        }
        
        // Butterworth low-pass filter design
        public static FilterCoefficients butterworthLowPass(double cutoffFreq, double sampleRate, int order) {
            double wc = 2 * Math.PI * cutoffFreq / sampleRate; // Digital cutoff frequency
            double wcPre = 2 * Math.tan(wc / 2); // Pre-warped frequency
            
            // For simplicity, implement first-order Butterworth
            if (order == 1) {
                double alpha = wcPre / (2 + wcPre);
                return new FilterCoefficients(
                    new double[]{alpha, alpha},
                    new double[]{1, alpha - 1}
                );
            } else {
                // Second-order approximation
                double k = Math.tan(wc / 2);
                double k2 = k * k;
                double sqrt2 = Math.sqrt(2);
                double norm = 1 / (1 + sqrt2 * k + k2);
                
                return new FilterCoefficients(
                    new double[]{k2 * norm, 2 * k2 * norm, k2 * norm},
                    new double[]{1, 2 * (k2 - 1) * norm, (1 - sqrt2 * k + k2) * norm}
                );
            }
        }
        
        public static FilterCoefficients butterworthHighPass(double cutoffFreq, double sampleRate, int order) {
            double wc = 2 * Math.PI * cutoffFreq / sampleRate;
            
            if (order == 1) {
                double alpha = 2 / (2 + 2 * Math.tan(wc / 2));
                return new FilterCoefficients(
                    new double[]{alpha, -alpha},
                    new double[]{1, alpha - 1}
                );
            } else {
                // Second-order approximation
                double k = Math.tan(wc / 2);
                double k2 = k * k;
                double sqrt2 = Math.sqrt(2);
                double norm = 1 / (1 + sqrt2 * k + k2);
                
                return new FilterCoefficients(
                    new double[]{norm, -2 * norm, norm},
                    new double[]{1, 2 * (k2 - 1) * norm, (1 - sqrt2 * k + k2) * norm}
                );
            }
        }
        
        public static FilterCoefficients butterworthBandPass(double lowFreq, double highFreq, 
                                                           double sampleRate, int order) {
            double w1 = 2 * Math.PI * lowFreq / sampleRate;
            double w2 = 2 * Math.PI * highFreq / sampleRate;
            double wc = Math.sqrt(w1 * w2); // Center frequency
            double bw = w2 - w1; // Bandwidth
            
            // Simplified second-order bandpass
            double r = 1 - 3 * bw;
            double k = (1 - 2 * r * Math.cos(wc) + r * r) / (2 - 2 * Math.cos(wc));
            double a1 = 2 * r * Math.cos(wc);
            double a2 = -r * r;
            double b0 = 1 - k;
            double b1 = 2 * (k - r) * Math.cos(wc);
            double b2 = r * r - k;
            
            return new FilterCoefficients(
                new double[]{b0, b1, b2},
                new double[]{1, a1, a2}
            );
        }
        
        // Moving average filter
        public static double[] movingAverage(double[] signal, int windowSize) {
            double[] output = new double[signal.length];
            
            for (int i = 0; i < signal.length; i++) {
                double sum = 0;
                int count = 0;
                
                for (int j = Math.max(0, i - windowSize / 2); 
                     j <= Math.min(signal.length - 1, i + windowSize / 2); j++) {
                    sum += signal[j];
                    count++;
                }
                
                output[i] = sum / count;
            }
            
            return output;
        }
        
        // Median filter for noise reduction
        public static double[] medianFilter(double[] signal, int windowSize) {
            double[] output = new double[signal.length];
            
            for (int i = 0; i < signal.length; i++) {
                List<Double> window = new ArrayList<>();
                
                for (int j = Math.max(0, i - windowSize / 2); 
                     j <= Math.min(signal.length - 1, i + windowSize / 2); j++) {
                    window.add(signal[j]);
                }
                
                Collections.sort(window);
                output[i] = window.get(window.size() / 2);
            }
            
            return output;
        }
        
        // Savitzky-Golay smoothing filter
        public static double[] savitzkyGolayFilter(double[] signal, int windowSize, int polynomialOrder) {
            if (windowSize % 2 == 0) windowSize++; // Ensure odd window size
            int halfWindow = windowSize / 2;
            
            // Compute Savitzky-Golay coefficients
            double[] coefficients = computeSavitzkyGolayCoefficients(windowSize, polynomialOrder);
            
            double[] output = new double[signal.length];
            
            for (int i = 0; i < signal.length; i++) {
                double sum = 0;
                
                for (int j = -halfWindow; j <= halfWindow; j++) {
                    int index = i + j;
                    if (index >= 0 && index < signal.length) {
                        sum += signal[index] * coefficients[j + halfWindow];
                    }
                }
                
                output[i] = sum;
            }
            
            return output;
        }
        
        private static double[] computeSavitzkyGolayCoefficients(int windowSize, int order) {
            // Simplified computation for commonly used parameters
            // Full implementation would require matrix operations
            
            if (windowSize == 5 && order == 2) {
                // Quadratic smoothing, 5-point window
                return new double[]{-3.0/35, 12.0/35, 17.0/35, 12.0/35, -3.0/35};
            } else if (windowSize == 7 && order == 2) {
                // Quadratic smoothing, 7-point window
                return new double[]{-2.0/21, 3.0/21, 6.0/21, 7.0/21, 6.0/21, 3.0/21, -2.0/21};
            } else {
                // Fall back to moving average
                double[] coeffs = new double[windowSize];
                Arrays.fill(coeffs, 1.0 / windowSize);
                return coeffs;
            }
        }
    }
    
    /**
     * Audio Signal Processing
     */
    public static class AudioProcessing {
        
        public static class AudioFrame {
            public final double[] samples;
            public final double sampleRate;
            public final int frameNumber;
            
            public AudioFrame(double[] samples, double sampleRate, int frameNumber) {
                this.samples = samples.clone();
                this.sampleRate = sampleRate;
                this.frameNumber = frameNumber;
            }
            
            public double getEnergy() {
                double energy = 0;
                for (double sample : samples) {
                    energy += sample * sample;
                }
                return energy;
            }
            
            public double getRMS() {
                return Math.sqrt(getEnergy() / samples.length);
            }
        }
        
        // Audio effects
        public static double[] reverb(double[] signal, double delay, double decay, double mix) {
            int delaySamples = (int) (delay * 44100); // Assume 44.1kHz sample rate
            double[] output = new double[signal.length + delaySamples];
            
            // Copy original signal
            System.arraycopy(signal, 0, output, 0, signal.length);
            
            // Add delayed and decayed signal
            for (int i = 0; i < signal.length; i++) {
                int delayedIndex = i + delaySamples;
                if (delayedIndex < output.length) {
                    output[delayedIndex] += signal[i] * decay;
                }
            }
            
            // Mix with original
            for (int i = 0; i < signal.length; i++) {
                output[i] = signal[i] * (1 - mix) + output[i] * mix;
            }
            
            return Arrays.copyOf(output, signal.length);
        }
        
        public static double[] echo(double[] signal, double delay, double feedback, double mix) {
            int delaySamples = (int) (delay * 44100);
            double[] delayLine = new double[delaySamples];
            double[] output = new double[signal.length];
            int delayIndex = 0;
            
            for (int i = 0; i < signal.length; i++) {
                double delayedSample = delayLine[delayIndex];
                double inputSample = signal[i] + delayedSample * feedback;
                
                output[i] = signal[i] * (1 - mix) + delayedSample * mix;
                delayLine[delayIndex] = inputSample;
                
                delayIndex = (delayIndex + 1) % delaySamples;
            }
            
            return output;
        }
        
        public static double[] distortion(double[] signal, double gain, double threshold) {
            double[] output = new double[signal.length];
            
            for (int i = 0; i < signal.length; i++) {
                double sample = signal[i] * gain;
                
                // Soft clipping
                if (sample > threshold) {
                    sample = threshold + (sample - threshold) / (1 + Math.pow((sample - threshold) / threshold, 2));
                } else if (sample < -threshold) {
                    sample = -threshold + (sample + threshold) / (1 + Math.pow((-sample - threshold) / threshold, 2));
                }
                
                output[i] = sample;
            }
            
            return output;
        }
        
        public static double[] chorus(double[] signal, double rate, double depth, double delay, double feedback) {
            int maxDelay = (int) ((delay + depth) * 44100);
            double[] delayLine = new double[maxDelay];
            double[] output = new double[signal.length];
            int writeIndex = 0;
            double lfoPhase = 0;
            
            for (int i = 0; i < signal.length; i++) {
                // LFO for modulation
                double lfo = Math.sin(lfoPhase) * depth + delay;
                int readDelay = (int) (lfo * 44100);
                int readIndex = (writeIndex - readDelay + maxDelay) % maxDelay;
                
                double delayedSample = delayLine[readIndex];
                double inputSample = signal[i] + delayedSample * feedback;
                
                output[i] = (signal[i] + delayedSample) * 0.5;
                delayLine[writeIndex] = inputSample;
                
                writeIndex = (writeIndex + 1) % maxDelay;
                lfoPhase += 2 * Math.PI * rate / 44100;
            }
            
            return output;
        }
        
        // Pitch detection using autocorrelation
        public static double detectPitch(double[] signal, double sampleRate, double minFreq, double maxFreq) {
            int minPeriod = (int) (sampleRate / maxFreq);
            int maxPeriod = (int) (sampleRate / minFreq);
            
            double[] autocorr = autocorrelation(signal);
            
            // Find peak in autocorrelation within valid period range
            double maxCorr = 0;
            int bestPeriod = minPeriod;
            
            for (int period = minPeriod; period <= Math.min(maxPeriod, autocorr.length / 2); period++) {
                if (autocorr[period] > maxCorr) {
                    maxCorr = autocorr[period];
                    bestPeriod = period;
                }
            }
            
            return sampleRate / bestPeriod;
        }
        
        private static double[] autocorrelation(double[] signal) {
            int n = signal.length;
            double[] autocorr = new double[n];
            
            for (int lag = 0; lag < n; lag++) {
                double sum = 0;
                for (int i = 0; i < n - lag; i++) {
                    sum += signal[i] * signal[i + lag];
                }
                autocorr[lag] = sum;
            }
            
            // Normalize
            if (autocorr[0] > 0) {
                for (int i = 0; i < n; i++) {
                    autocorr[i] /= autocorr[0];
                }
            }
            
            return autocorr;
        }
        
        // Voice Activity Detection
        public static boolean[] voiceActivityDetection(double[] signal, int frameSize, int hopSize, 
                                                     double energyThreshold, double zcThreshold) {
            int numFrames = (signal.length - frameSize) / hopSize + 1;
            boolean[] vad = new boolean[numFrames];
            
            for (int frame = 0; frame < numFrames; frame++) {
                int start = frame * hopSize;
                int end = Math.min(start + frameSize, signal.length);
                
                // Compute frame energy
                double energy = 0;
                for (int i = start; i < end; i++) {
                    energy += signal[i] * signal[i];
                }
                energy = Math.sqrt(energy / (end - start));
                
                // Compute zero crossing rate
                int zeroCrossings = 0;
                for (int i = start + 1; i < end; i++) {
                    if ((signal[i] >= 0) != (signal[i - 1] >= 0)) {
                        zeroCrossings++;
                    }
                }
                double zcr = (double) zeroCrossings / (end - start);
                
                // Voice activity decision
                vad[frame] = energy > energyThreshold && zcr < zcThreshold;
            }
            
            return vad;
        }
        
        // Spectral centroid for timbre analysis
        public static double spectralCentroid(Complex[] spectrum, double sampleRate) {
            double[] magnitude = FourierTransform.magnitudeSpectrum(spectrum);
            double weightedSum = 0;
            double magnitudeSum = 0;
            
            for (int k = 0; k < magnitude.length / 2; k++) {
                double frequency = k * sampleRate / magnitude.length;
                weightedSum += frequency * magnitude[k];
                magnitudeSum += magnitude[k];
            }
            
            return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
        }
        
        // Mel-frequency cepstral coefficients (simplified)
        public static double[] mfcc(double[] signal, double sampleRate, int numCoefficients) {
            // Convert to frequency domain
            Complex[] signalComplex = new Complex[signal.length];
            for (int i = 0; i < signal.length; i++) {
                signalComplex[i] = new Complex(signal[i]);
            }
            
            Complex[] spectrum = FourierTransform.fft(signalComplex);
            double[] powerSpectrum = FourierTransform.powerSpectrum(spectrum);
            
            // Apply mel filterbank
            double[] melSpectrum = applyMelFilterbank(powerSpectrum, sampleRate, 26);
            
            // Log compression
            for (int i = 0; i < melSpectrum.length; i++) {
                melSpectrum[i] = Math.log(melSpectrum[i] + 1e-10);
            }
            
            // DCT to get cepstral coefficients
            double[] mfccCoeffs = FourierTransform.dct(melSpectrum);
            
            return Arrays.copyOf(mfccCoeffs, numCoefficients);
        }
        
        private static double[] applyMelFilterbank(double[] powerSpectrum, double sampleRate, int numFilters) {
            int nfft = powerSpectrum.length;
            double[] melFiltered = new double[numFilters];
            
            // Mel scale conversion
            double melMax = 2595 * Math.log10(1 + (sampleRate / 2) / 700);
            double melMin = 2595 * Math.log10(1 + 300 / 700);
            
            // Create triangular filters
            for (int m = 0; m < numFilters; m++) {
                double melCenter = melMin + (m + 1) * (melMax - melMin) / (numFilters + 1);
                double freqCenter = 700 * (Math.pow(10, melCenter / 2595) - 1);
                int binCenter = (int) (freqCenter * nfft / sampleRate);
                
                double sum = 0;
                for (int k = Math.max(0, binCenter - 10); k < Math.min(nfft / 2, binCenter + 10); k++) {
                    double weight = 1.0 - Math.abs(k - binCenter) / 10.0;
                    if (weight > 0) {
                        sum += powerSpectrum[k] * weight;
                    }
                }
                
                melFiltered[m] = sum;
            }
            
            return melFiltered;
        }
    }
    
    /**
     * Adaptive Signal Processing
     */
    public static class AdaptiveFilters {
        
        public static class LMSFilter {
            private final double[] weights;
            private final double[] inputBuffer;
            private final double learningRate;
            private int bufferIndex;
            
            public LMSFilter(int numTaps, double learningRate) {
                this.weights = new double[numTaps];
                this.inputBuffer = new double[numTaps];
                this.learningRate = learningRate;
                this.bufferIndex = 0;
            }
            
            public double filter(double input, double desired) {
                // Update input buffer
                inputBuffer[bufferIndex] = input;
                bufferIndex = (bufferIndex + 1) % inputBuffer.length;
                
                // Compute output
                double output = 0;
                for (int i = 0; i < weights.length; i++) {
                    int index = (bufferIndex - 1 - i + inputBuffer.length) % inputBuffer.length;
                    output += weights[i] * inputBuffer[index];
                }
                
                // Compute error and update weights
                double error = desired - output;
                for (int i = 0; i < weights.length; i++) {
                    int index = (bufferIndex - 1 - i + inputBuffer.length) % inputBuffer.length;
                    weights[i] += learningRate * error * inputBuffer[index];
                }
                
                return output;
            }
            
            public double[] getWeights() {
                return weights.clone();
            }
        }
        
        public static class RLSFilter {
            private final double[][] P; // Inverse correlation matrix
            private final double[] weights;
            private final double[] inputBuffer;
            private final double forgettingFactor;
            private int bufferIndex;
            
            public RLSFilter(int numTaps, double forgettingFactor, double delta) {
                this.weights = new double[numTaps];
                this.inputBuffer = new double[numTaps];
                this.forgettingFactor = forgettingFactor;
                this.bufferIndex = 0;
                
                // Initialize P matrix
                this.P = new double[numTaps][numTaps];
                for (int i = 0; i < numTaps; i++) {
                    P[i][i] = 1.0 / delta;
                }
            }
            
            public double filter(double input, double desired) {
                // Update input buffer
                inputBuffer[bufferIndex] = input;
                bufferIndex = (bufferIndex + 1) % inputBuffer.length;
                
                // Create input vector
                double[] x = new double[weights.length];
                for (int i = 0; i < weights.length; i++) {
                    int index = (bufferIndex - 1 - i + inputBuffer.length) % inputBuffer.length;
                    x[i] = inputBuffer[index];
                }
                
                // Compute output
                double output = 0;
                for (int i = 0; i < weights.length; i++) {
                    output += weights[i] * x[i];
                }
                
                // RLS update
                double[] Px = matrixVectorMultiply(P, x);
                double denominator = forgettingFactor + vectorDotProduct(x, Px);
                
                double error = desired - output;
                
                // Update weights
                for (int i = 0; i < weights.length; i++) {
                    weights[i] += error * Px[i] / denominator;
                }
                
                // Update P matrix
                for (int i = 0; i < P.length; i++) {
                    for (int j = 0; j < P[0].length; j++) {
                        P[i][j] = (P[i][j] - Px[i] * Px[j] / denominator) / forgettingFactor;
                    }
                }
                
                return output;
            }
            
            private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
                double[] result = new double[matrix.length];
                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < vector.length; j++) {
                        result[i] += matrix[i][j] * vector[j];
                    }
                }
                return result;
            }
            
            private double vectorDotProduct(double[] a, double[] b) {
                double sum = 0;
                for (int i = 0; i < a.length; i++) {
                    sum += a[i] * b[i];
                }
                return sum;
            }
        }
        
        // Normalized LMS (NLMS)
        public static class NLMSFilter {
            private final double[] weights;
            private final double[] inputBuffer;
            private final double learningRate;
            private final double regularization;
            private int bufferIndex;
            
            public NLMSFilter(int numTaps, double learningRate, double regularization) {
                this.weights = new double[numTaps];
                this.inputBuffer = new double[numTaps];
                this.learningRate = learningRate;
                this.regularization = regularization;
                this.bufferIndex = 0;
            }
            
            public double filter(double input, double desired) {
                // Update input buffer
                inputBuffer[bufferIndex] = input;
                bufferIndex = (bufferIndex + 1) % inputBuffer.length;
                
                // Compute output
                double output = 0;
                double inputPower = 0;
                
                for (int i = 0; i < weights.length; i++) {
                    int index = (bufferIndex - 1 - i + inputBuffer.length) % inputBuffer.length;
                    output += weights[i] * inputBuffer[index];
                    inputPower += inputBuffer[index] * inputBuffer[index];
                }
                
                // Compute error and normalized step size
                double error = desired - output;
                double stepSize = learningRate / (inputPower + regularization);
                
                // Update weights
                for (int i = 0; i < weights.length; i++) {
                    int index = (bufferIndex - 1 - i + inputBuffer.length) % inputBuffer.length;
                    weights[i] += stepSize * error * inputBuffer[index];
                }
                
                return output;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Signal Processing Algorithms Demo:");
        System.out.println("==================================");
        
        // Generate test signals
        double sampleRate = 1000;
        int signalLength = 512;
        double[] testSignal = generateTestSignal(signalLength, sampleRate);
        double[] noisySignal = addNoise(testSignal, 0.1);
        
        // Fourier Transform
        System.out.println("1. Fourier Transform:");
        Complex[] signalComplex = new Complex[signalLength];
        for (int i = 0; i < signalLength; i++) {
            signalComplex[i] = new Complex(testSignal[i]);
        }
        
        Complex[] spectrum = FourierTransform.fft(signalComplex);
        double[] magnitude = FourierTransform.magnitudeSpectrum(spectrum);
        double[] power = FourierTransform.powerSpectrum(spectrum);
        
        System.out.printf("FFT computed: %d frequency bins\n", spectrum.length);
        System.out.printf("Peak magnitude: %.3f at bin %d\n", 
                         Arrays.stream(magnitude).max().orElse(0),
                         findMaxIndex(magnitude));
        
        // STFT
        Complex[][] spectrogram = FourierTransform.stft(testSignal, 64, 32);
        System.out.printf("STFT computed: %d time frames, %d frequency bins\n", 
                         spectrogram.length, spectrogram[0].length);
        
        // DCT
        double[] dctCoeffs = FourierTransform.dct(Arrays.copyOf(testSignal, 64));
        System.out.printf("DCT coefficients computed: %d coefficients\n", dctCoeffs.length);
        
        // Digital Filters
        System.out.println("\n2. Digital Filters:");
        
        // Low-pass filter
        DigitalFilters.FilterCoefficients lpf = DigitalFilters.butterworthLowPass(50, sampleRate, 2);
        double[] filteredSignal = DigitalFilters.iirFilter(noisySignal, lpf);
        System.out.printf("Low-pass filtered signal: %.3f energy reduction\n",
                         1 - computeEnergy(filteredSignal) / computeEnergy(noisySignal));
        
        // Moving average
        double[] smoothed = DigitalFilters.movingAverage(noisySignal, 10);
        System.out.printf("Moving average applied: %.3f smoothing factor\n",
                         computeEnergy(noisySignal) / computeEnergy(smoothed));
        
        // Median filter
        double[] medianFiltered = DigitalFilters.medianFilter(noisySignal, 5);
        System.out.printf("Median filter applied: %d samples processed\n", medianFiltered.length);
        
        // Savitzky-Golay
        double[] sgFiltered = DigitalFilters.savitzkyGolayFilter(noisySignal, 7, 2);
        System.out.printf("Savitzky-Golay filter applied: %.3f smoothness improvement\n",
                         computeSmoothness(sgFiltered) / computeSmoothness(noisySignal));
        
        // Audio Processing
        System.out.println("\n3. Audio Processing:");
        
        // Audio effects
        double[] reverbed = AudioProcessing.reverb(testSignal, 0.1, 0.3, 0.2);
        System.out.printf("Reverb applied: %.1f%% wet signal\n", 20.0);
        
        double[] echoed = AudioProcessing.echo(testSignal, 0.2, 0.4, 0.3);
        System.out.printf("Echo applied: %.1f%% feedback\n", 40.0);
        
        double[] distorted = AudioProcessing.distortion(testSignal, 2.0, 0.5);
        System.out.printf("Distortion applied: 2x gain, 0.5 threshold\n");
        
        // Pitch detection
        double[] pitchSignal = generateSineWave(440, 0.5, sampleRate); // 440 Hz tone
        double detectedPitch = AudioProcessing.detectPitch(pitchSignal, sampleRate, 100, 1000);
        System.out.printf("Pitch detection: %.1f Hz (expected: 440 Hz)\n", detectedPitch);
        
        // Voice Activity Detection
        boolean[] vad = AudioProcessing.voiceActivityDetection(testSignal, 64, 32, 0.01, 0.1);
        int activeFrames = 0;
        for (boolean voice : vad) {
            if (voice) activeFrames++;
        }
        System.out.printf("Voice activity: %d/%d frames active\n", activeFrames, vad.length);
        
        // Spectral centroid
        double centroid = AudioProcessing.spectralCentroid(spectrum, sampleRate);
        System.out.printf("Spectral centroid: %.1f Hz\n", centroid);
        
        // MFCC
        double[] mfccCoeffs = AudioProcessing.mfcc(Arrays.copyOf(testSignal, 256), sampleRate, 13);
        System.out.printf("MFCC computed: %d coefficients\n", mfccCoeffs.length);
        
        // Adaptive Filters
        System.out.println("\n4. Adaptive Filters:");
        
        // LMS Filter
        AdaptiveFilters.LMSFilter lms = new AdaptiveFilters.LMSFilter(10, 0.01);
        double[] lmsOutput = new double[testSignal.length];
        
        for (int i = 0; i < testSignal.length; i++) {
            // Use noisy signal as input, clean signal as desired
            lmsOutput[i] = lms.filter(noisySignal[i], testSignal[i]);
        }
        
        double lmsError = computeMSE(testSignal, lmsOutput);
        System.out.printf("LMS filter: MSE = %.6f\n", lmsError);
        
        // RLS Filter
        AdaptiveFilters.RLSFilter rls = new AdaptiveFilters.RLSFilter(10, 0.99, 0.1);
        double[] rlsOutput = new double[testSignal.length];
        
        for (int i = 0; i < testSignal.length; i++) {
            rlsOutput[i] = rls.filter(noisySignal[i], testSignal[i]);
        }
        
        double rlsError = computeMSE(testSignal, rlsOutput);
        System.out.printf("RLS filter: MSE = %.6f\n", rlsError);
        
        // NLMS Filter
        AdaptiveFilters.NLMSFilter nlms = new AdaptiveFilters.NLMSFilter(10, 0.5, 0.001);
        double[] nlmsOutput = new double[testSignal.length];
        
        for (int i = 0; i < testSignal.length; i++) {
            nlmsOutput[i] = nlms.filter(noisySignal[i], testSignal[i]);
        }
        
        double nlmsError = computeMSE(testSignal, nlmsOutput);
        System.out.printf("NLMS filter: MSE = %.6f\n", nlmsError);
        
        System.out.println("\nSignal processing demonstration completed!");
        System.out.println("Algorithms demonstrated:");
        System.out.println("- Fourier transforms: FFT, IFFT, STFT, DCT");
        System.out.println("- Digital filters: Butterworth, FIR, IIR, moving average, median, Savitzky-Golay");
        System.out.println("- Audio processing: Reverb, echo, distortion, chorus, pitch detection, VAD, MFCC");
        System.out.println("- Adaptive filters: LMS, RLS, NLMS");
    }
    
    private static double[] generateTestSignal(int length, double sampleRate) {
        double[] signal = new double[length];
        for (int i = 0; i < length; i++) {
            double t = i / sampleRate;
            signal[i] = Math.sin(2 * Math.PI * 50 * t) + 0.5 * Math.sin(2 * Math.PI * 120 * t);
        }
        return signal;
    }
    
    private static double[] generateSineWave(double frequency, double duration, double sampleRate) {
        int length = (int) (duration * sampleRate);
        double[] signal = new double[length];
        for (int i = 0; i < length; i++) {
            double t = i / sampleRate;
            signal[i] = Math.sin(2 * Math.PI * frequency * t);
        }
        return signal;
    }
    
    private static double[] addNoise(double[] signal, double noiseLevel) {
        Random random = new Random(42);
        double[] noisy = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            noisy[i] = signal[i] + noiseLevel * random.nextGaussian();
        }
        return noisy;
    }
    
    private static double computeEnergy(double[] signal) {
        double energy = 0;
        for (double sample : signal) {
            energy += sample * sample;
        }
        return energy;
    }
    
    private static double computeSmoothness(double[] signal) {
        double smoothness = 0;
        for (int i = 1; i < signal.length; i++) {
            double diff = signal[i] - signal[i-1];
            smoothness += diff * diff;
        }
        return smoothness;
    }
    
    private static double computeMSE(double[] reference, double[] signal) {
        double mse = 0;
        for (int i = 0; i < Math.min(reference.length, signal.length); i++) {
            double error = reference[i] - signal[i];
            mse += error * error;
        }
        return mse / Math.min(reference.length, signal.length);
    }
    
    private static int findMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
