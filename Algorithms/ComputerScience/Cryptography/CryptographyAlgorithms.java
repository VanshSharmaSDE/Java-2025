package Algorithms.ComputerScience.Cryptography;

import java.math.BigInteger;
import java.security.SecureRandom;

/**
 * Cryptographic Algorithms
 */
public class CryptographyAlgorithms {
    
    /**
     * Caesar Cipher - Simple substitution cipher
     */
    public static class CaesarCipher {
        
        /**
         * Encrypt text using Caesar cipher
         * @param text Plain text
         * @param shift Shift amount
         * @return Encrypted text
         */
        public static String encrypt(String text, int shift) {
            StringBuilder result = new StringBuilder();
            
            for (char character : text.toCharArray()) {
                if (Character.isLetter(character)) {
                    char base = Character.isLowerCase(character) ? 'a' : 'A';
                    result.append((char) ((character - base + shift) % 26 + base));
                } else {
                    result.append(character);
                }
            }
            
            return result.toString();
        }
        
        /**
         * Decrypt text using Caesar cipher
         * @param cipherText Encrypted text
         * @param shift Shift amount
         * @return Decrypted text
         */
        public static String decrypt(String cipherText, int shift) {
            return encrypt(cipherText, 26 - shift);
        }
    }
    
    /**
     * Vigenère Cipher - Polyalphabetic substitution cipher
     */
    public static class VigenereCipher {
        
        /**
         * Encrypt text using Vigenère cipher
         * @param text Plain text
         * @param key Encryption key
         * @return Encrypted text
         */
        public static String encrypt(String text, String key) {
            StringBuilder result = new StringBuilder();
            key = key.toLowerCase();
            int keyIndex = 0;
            
            for (char character : text.toCharArray()) {
                if (Character.isLetter(character)) {
                    char base = Character.isLowerCase(character) ? 'a' : 'A';
                    int shift = key.charAt(keyIndex % key.length()) - 'a';
                    result.append((char) ((character - base + shift) % 26 + base));
                    keyIndex++;
                } else {
                    result.append(character);
                }
            }
            
            return result.toString();
        }
        
        /**
         * Decrypt text using Vigenère cipher
         * @param cipherText Encrypted text
         * @param key Decryption key
         * @return Decrypted text
         */
        public static String decrypt(String cipherText, String key) {
            StringBuilder result = new StringBuilder();
            key = key.toLowerCase();
            int keyIndex = 0;
            
            for (char character : cipherText.toCharArray()) {
                if (Character.isLetter(character)) {
                    char base = Character.isLowerCase(character) ? 'a' : 'A';
                    int shift = key.charAt(keyIndex % key.length()) - 'a';
                    result.append((char) ((character - base - shift + 26) % 26 + base));
                    keyIndex++;
                } else {
                    result.append(character);
                }
            }
            
            return result.toString();
        }
    }
    
    /**
     * RSA Algorithm Implementation (Simplified)
     */
    public static class RSA {
        private BigInteger n, d, e;
        private int bitLength = 1024;
        
        /**
         * Generate RSA key pair
         */
        public void generateKeys() {
            SecureRandom random = new SecureRandom();
            
            // Generate two large prime numbers
            BigInteger p = BigInteger.probablePrime(bitLength / 2, random);
            BigInteger q = BigInteger.probablePrime(bitLength / 2, random);
            
            // Calculate n = p * q
            n = p.multiply(q);
            
            // Calculate φ(n) = (p-1)(q-1)
            BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));
            
            // Choose e (commonly 65537)
            e = BigInteger.valueOf(65537);
            
            // Calculate d = e^(-1) mod φ(n)
            d = e.modInverse(phi);
        }
        
        /**
         * Encrypt message
         * @param message Message as BigInteger
         * @return Encrypted message
         */
        public BigInteger encrypt(BigInteger message) {
            return message.modPow(e, n);
        }
        
        /**
         * Decrypt message
         * @param ciphertext Encrypted message
         * @return Decrypted message
         */
        public BigInteger decrypt(BigInteger ciphertext) {
            return ciphertext.modPow(d, n);
        }
        
        public BigInteger getN() { return n; }
        public BigInteger getE() { return e; }
        public BigInteger getD() { return d; }
    }
    
    /**
     * Diffie-Hellman Key Exchange
     */
    public static class DiffieHellman {
        private static final BigInteger P = new BigInteger("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1" +
                "29024E088A67CC74020BBEA63B139B22514A08798E3404DD" +
                "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245" +
                "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" +
                "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D" +
                "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" +
                "83655D23DCA3AD961C62F356208552BB9ED529077096966D" +
                "670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF", 16);
        private static final BigInteger G = BigInteger.valueOf(2);
        
        private BigInteger privateKey;
        private BigInteger publicKey;
        
        /**
         * Generate key pair
         */
        public void generateKeys() {
            SecureRandom random = new SecureRandom();
            privateKey = new BigInteger(1023, random);
            publicKey = G.modPow(privateKey, P);
        }
        
        /**
         * Calculate shared secret
         * @param otherPublicKey Other party's public key
         * @return Shared secret
         */
        public BigInteger calculateSharedSecret(BigInteger otherPublicKey) {
            return otherPublicKey.modPow(privateKey, P);
        }
        
        public BigInteger getPublicKey() { return publicKey; }
    }
    
    /**
     * Hash function (Simple implementation for demonstration)
     */
    public static class SimpleHash {
        
        /**
         * Simple hash function (not cryptographically secure)
         * @param input Input string
         * @return Hash value
         */
        public static int hash(String input) {
            int hash = 0;
            for (char c : input.toCharArray()) {
                hash = hash * 31 + c;
            }
            return Math.abs(hash);
        }
        
        /**
         * MD5-like hash (simplified)
         * @param input Input string
         * @return Hash as hex string
         */
        public static String md5Like(String input) {
            int hash = hash(input);
            return String.format("%08x", hash);
        }
    }
    
    /**
     * Linear Congruential Generator (Simple PRNG)
     */
    public static class SimpleRNG {
        private long seed;
        private static final long A = 1664525L;
        private static final long C = 1013904223L;
        private static final long M = (1L << 32);
        
        public SimpleRNG(long seed) {
            this.seed = seed;
        }
        
        /**
         * Generate next random number
         * @return Random number
         */
        public int next() {
            seed = (A * seed + C) % M;
            return (int) (seed & 0x7FFFFFFF);
        }
        
        /**
         * Generate random number in range [0, max)
         * @param max Maximum value (exclusive)
         * @return Random number
         */
        public int next(int max) {
            return next() % max;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Cryptographic Algorithms:");
        System.out.println("=========================");
        
        // Caesar Cipher
        String plaintext = "Hello World";
        int shift = 3;
        String encrypted = CaesarCipher.encrypt(plaintext, shift);
        String decrypted = CaesarCipher.decrypt(encrypted, shift);
        
        System.out.println("Caesar Cipher:");
        System.out.println("Plaintext: " + plaintext);
        System.out.println("Encrypted: " + encrypted);
        System.out.println("Decrypted: " + decrypted);
        
        // Vigenère Cipher
        String key = "KEY";
        String vEncrypted = VigenereCipher.encrypt(plaintext, key);
        String vDecrypted = VigenereCipher.decrypt(vEncrypted, key);
        
        System.out.println("\nVigenère Cipher:");
        System.out.println("Key: " + key);
        System.out.println("Plaintext: " + plaintext);
        System.out.println("Encrypted: " + vEncrypted);
        System.out.println("Decrypted: " + vDecrypted);
        
        // Simple Hash
        String message = "This is a test message";
        int hashValue = SimpleHash.hash(message);
        String hexHash = SimpleHash.md5Like(message);
        
        System.out.println("\nSimple Hash:");
        System.out.println("Message: " + message);
        System.out.println("Hash: " + hashValue);
        System.out.println("Hex Hash: " + hexHash);
        
        // Diffie-Hellman Key Exchange
        System.out.println("\nDiffie-Hellman Key Exchange:");
        DiffieHellman alice = new DiffieHellman();
        DiffieHellman bob = new DiffieHellman();
        
        alice.generateKeys();
        bob.generateKeys();
        
        BigInteger aliceSecret = alice.calculateSharedSecret(bob.getPublicKey());
        BigInteger bobSecret = bob.calculateSharedSecret(alice.getPublicKey());
        
        System.out.println("Alice's shared secret: " + aliceSecret.toString(16).substring(0, 16) + "...");
        System.out.println("Bob's shared secret: " + bobSecret.toString(16).substring(0, 16) + "...");
        System.out.println("Secrets match: " + aliceSecret.equals(bobSecret));
        
        // Simple RNG
        System.out.println("\nSimple Random Number Generator:");
        SimpleRNG rng = new SimpleRNG(12345);
        System.out.print("Random numbers: ");
        for (int i = 0; i < 10; i++) {
            System.out.print(rng.next(100) + " ");
        }
        System.out.println();
        
        // Note: RSA example omitted from main due to key generation time
        System.out.println("\nNote: RSA implementation available but key generation may take time for demo.");
    }
}
