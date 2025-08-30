package Algorithms.CryptographyAndSecurity;

import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.*;
import java.nio.charset.StandardCharsets;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

/**
 * Comprehensive Cryptography and Security Algorithms
 * Modern encryption, digital signatures, and security protocols
 */
public class CryptographyAlgorithms {
    
    /**
     * RSA Public Key Cryptography Implementation
     */
    public static class RSACryptography {
        private BigInteger n, e, d;
        private static final SecureRandom random = new SecureRandom();
        
        public static class RSAKeyPair {
            public final BigInteger n, e, d;
            
            public RSAKeyPair(BigInteger n, BigInteger e, BigInteger d) {
                this.n = n; this.e = e; this.d = d;
            }
            
            public String toString() {
                return String.format("Public Key: (n=%s, e=%s)\nPrivate Key: (n=%s, d=%s)", 
                                   n.toString(16), e.toString(16), n.toString(16), d.toString(16));
            }
        }
        
        public static RSAKeyPair generateKeyPair(int bitLength) {
            // Generate two large primes
            BigInteger p = BigInteger.probablePrime(bitLength / 2, random);
            BigInteger q = BigInteger.probablePrime(bitLength / 2, random);
            
            // Compute n = p * q
            BigInteger n = p.multiply(q);
            
            // Compute φ(n) = (p-1)(q-1)
            BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));
            
            // Choose e such that gcd(e, φ(n)) = 1
            BigInteger e = BigInteger.valueOf(65537); // Common choice
            
            // Compute d = e^(-1) mod φ(n)
            BigInteger d = e.modInverse(phi);
            
            return new RSAKeyPair(n, e, d);
        }
        
        public static BigInteger encrypt(BigInteger message, BigInteger e, BigInteger n) {
            return message.modPow(e, n);
        }
        
        public static BigInteger decrypt(BigInteger ciphertext, BigInteger d, BigInteger n) {
            return ciphertext.modPow(d, n);
        }
        
        public static String encryptString(String plaintext, BigInteger e, BigInteger n) {
            byte[] bytes = plaintext.getBytes(StandardCharsets.UTF_8);
            StringBuilder result = new StringBuilder();
            
            for (byte b : bytes) {
                BigInteger m = BigInteger.valueOf(b & 0xFF);
                BigInteger c = encrypt(m, e, n);
                result.append(c.toString(16)).append(" ");
            }
            
            return result.toString().trim();
        }
        
        public static String decryptString(String ciphertext, BigInteger d, BigInteger n) {
            String[] parts = ciphertext.split(" ");
            StringBuilder result = new StringBuilder();
            
            for (String part : parts) {
                BigInteger c = new BigInteger(part, 16);
                BigInteger m = decrypt(c, d, n);
                result.append((char) m.intValue());
            }
            
            return result.toString();
        }
    }
    
    /**
     * Elliptic Curve Cryptography (ECC)
     */
    public static class EllipticCurveCryptography {
        
        public static class Point {
            public final BigInteger x, y;
            public final boolean isInfinity;
            
            public Point(BigInteger x, BigInteger y) {
                this.x = x; this.y = y; this.isInfinity = false;
            }
            
            public Point() { // Point at infinity
                this.x = null; this.y = null; this.isInfinity = true;
            }
            
            public boolean equals(Object obj) {
                if (!(obj instanceof Point)) return false;
                Point other = (Point) obj;
                if (isInfinity && other.isInfinity) return true;
                if (isInfinity || other.isInfinity) return false;
                return x.equals(other.x) && y.equals(other.y);
            }
            
            public String toString() {
                return isInfinity ? "O (infinity)" : String.format("(%s, %s)", x, y);
            }
        }
        
        public static class EllipticCurve {
            public final BigInteger a, b, p; // y² = x³ + ax + b (mod p)
            
            public EllipticCurve(BigInteger a, BigInteger b, BigInteger p) {
                this.a = a; this.b = b; this.p = p;
            }
            
            public boolean isOnCurve(Point point) {
                if (point.isInfinity) return true;
                
                BigInteger left = point.y.pow(2).mod(p);
                BigInteger right = point.x.pow(3).add(a.multiply(point.x)).add(b).mod(p);
                
                return left.equals(right);
            }
            
            public Point add(Point p1, Point p2) {
                if (p1.isInfinity) return p2;
                if (p2.isInfinity) return p1;
                
                if (p1.x.equals(p2.x)) {
                    if (p1.y.equals(p2.y)) {
                        return doublePoint(p1);
                    } else {
                        return new Point(); // Point at infinity
                    }
                }
                
                // Calculate slope
                BigInteger numerator = p2.y.subtract(p1.y);
                BigInteger denominator = p2.x.subtract(p1.x);
                BigInteger slope = numerator.multiply(denominator.modInverse(p)).mod(p);
                
                // Calculate new point
                BigInteger x3 = slope.pow(2).subtract(p1.x).subtract(p2.x).mod(p);
                BigInteger y3 = slope.multiply(p1.x.subtract(x3)).subtract(p1.y).mod(p);
                
                return new Point(x3, y3);
            }
            
            public Point doublePoint(Point point) {
                if (point.isInfinity) return point;
                if (point.y.equals(BigInteger.ZERO)) return new Point();
                
                // Calculate slope: (3x² + a) / (2y)
                BigInteger numerator = point.x.pow(2).multiply(BigInteger.valueOf(3)).add(a);
                BigInteger denominator = point.y.multiply(BigInteger.valueOf(2));
                BigInteger slope = numerator.multiply(denominator.modInverse(p)).mod(p);
                
                // Calculate new point
                BigInteger x3 = slope.pow(2).subtract(point.x.multiply(BigInteger.valueOf(2))).mod(p);
                BigInteger y3 = slope.multiply(point.x.subtract(x3)).subtract(point.y).mod(p);
                
                return new Point(x3, y3);
            }
            
            public Point multiply(BigInteger k, Point point) {
                if (k.equals(BigInteger.ZERO)) return new Point();
                if (k.equals(BigInteger.ONE)) return point;
                
                Point result = new Point();
                Point addend = point;
                
                while (k.compareTo(BigInteger.ZERO) > 0) {
                    if (k.testBit(0)) {
                        result = add(result, addend);
                    }
                    addend = doublePoint(addend);
                    k = k.shiftRight(1);
                }
                
                return result;
            }
        }
        
        public static class ECDSASignature {
            public final BigInteger r, s;
            
            public ECDSASignature(BigInteger r, BigInteger s) {
                this.r = r; this.s = s;
            }
            
            public String toString() {
                return String.format("ECDSA Signature: (r=%s, s=%s)", r.toString(16), s.toString(16));
            }
        }
        
        public static ECDSASignature sign(String message, BigInteger privateKey, EllipticCurve curve, Point basePoint, BigInteger order) {
            byte[] hash = sha256(message.getBytes(StandardCharsets.UTF_8));
            BigInteger z = new BigInteger(1, Arrays.copyOf(hash, Math.min(hash.length, order.bitLength() / 8)));
            
            BigInteger k, r, s;
            do {
                do {
                    k = new BigInteger(order.bitLength(), new SecureRandom()).mod(order.subtract(BigInteger.ONE)).add(BigInteger.ONE);
                    Point kG = curve.multiply(k, basePoint);
                    r = kG.x.mod(order);
                } while (r.equals(BigInteger.ZERO));
                
                BigInteger kInv = k.modInverse(order);
                s = kInv.multiply(z.add(privateKey.multiply(r))).mod(order);
            } while (s.equals(BigInteger.ZERO));
            
            return new ECDSASignature(r, s);
        }
        
        public static boolean verify(String message, ECDSASignature signature, Point publicKey, EllipticCurve curve, Point basePoint, BigInteger order) {
            byte[] hash = sha256(message.getBytes(StandardCharsets.UTF_8));
            BigInteger z = new BigInteger(1, Arrays.copyOf(hash, Math.min(hash.length, order.bitLength() / 8)));
            
            BigInteger sInv = signature.s.modInverse(order);
            BigInteger u1 = z.multiply(sInv).mod(order);
            BigInteger u2 = signature.r.multiply(sInv).mod(order);
            
            Point p1 = curve.multiply(u1, basePoint);
            Point p2 = curve.multiply(u2, publicKey);
            Point result = curve.add(p1, p2);
            
            return !result.isInfinity && result.x.mod(order).equals(signature.r);
        }
    }
    
    /**
     * Advanced Encryption Standard (AES) Implementation
     */
    public static class AESCryptography {
        private static final int[][] S_BOX = {
            {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
            {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
            // ... (S-box would be complete in full implementation)
        };
        
        private static final int[] RCON = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};
        
        public static byte[] encrypt(byte[] plaintext, byte[] key) {
            // Simplified AES implementation
            // In practice, use javax.crypto.Cipher
            try {
                Cipher cipher = Cipher.getInstance("AES");
                SecretKeySpec keySpec = new SecretKeySpec(key, "AES");
                cipher.init(Cipher.ENCRYPT_MODE, keySpec);
                return cipher.doFinal(plaintext);
            } catch (Exception e) {
                throw new RuntimeException("AES encryption failed", e);
            }
        }
        
        public static byte[] decrypt(byte[] ciphertext, byte[] key) {
            try {
                Cipher cipher = Cipher.getInstance("AES");
                SecretKeySpec keySpec = new SecretKeySpec(key, "AES");
                cipher.init(Cipher.DECRYPT_MODE, keySpec);
                return cipher.doFinal(ciphertext);
            } catch (Exception e) {
                throw new RuntimeException("AES decryption failed", e);
            }
        }
        
        public static byte[] generateKey() {
            try {
                KeyGenerator keyGen = KeyGenerator.getInstance("AES");
                keyGen.init(256);
                SecretKey secretKey = keyGen.generateKey();
                return secretKey.getEncoded();
            } catch (Exception e) {
                throw new RuntimeException("Key generation failed", e);
            }
        }
    }
    
    /**
     * Digital Signature Algorithm (DSA)
     */
    public static class DigitalSignatureAlgorithm {
        
        public static class DSAParameters {
            public final BigInteger p, q, g;
            
            public DSAParameters(BigInteger p, BigInteger q, BigInteger g) {
                this.p = p; this.q = q; this.g = g;
            }
        }
        
        public static class DSAKeyPair {
            public final BigInteger x, y; // private and public key
            public final DSAParameters params;
            
            public DSAKeyPair(BigInteger x, BigInteger y, DSAParameters params) {
                this.x = x; this.y = y; this.params = params;
            }
        }
        
        public static class DSASignature {
            public final BigInteger r, s;
            
            public DSASignature(BigInteger r, BigInteger s) {
                this.r = r; this.s = s;
            }
        }
        
        public static DSAParameters generateParameters(int L, int N) {
            SecureRandom random = new SecureRandom();
            
            // Generate q (N-bit prime)
            BigInteger q = BigInteger.probablePrime(N, random);
            
            // Generate p (L-bit prime such that q divides p-1)
            BigInteger p;
            do {
                BigInteger k = new BigInteger(L - N, random);
                p = k.multiply(q).add(BigInteger.ONE);
            } while (!p.isProbablePrime(50) || p.bitLength() != L);
            
            // Generate g
            BigInteger g;
            do {
                BigInteger h = new BigInteger(L - 1, random);
                g = h.modPow(p.subtract(BigInteger.ONE).divide(q), p);
            } while (g.equals(BigInteger.ONE));
            
            return new DSAParameters(p, q, g);
        }
        
        public static DSAKeyPair generateKeyPair(DSAParameters params) {
            SecureRandom random = new SecureRandom();
            
            // Generate private key x (0 < x < q)
            BigInteger x = new BigInteger(params.q.bitLength() - 1, random);
            
            // Generate public key y = g^x mod p
            BigInteger y = params.g.modPow(x, params.p);
            
            return new DSAKeyPair(x, y, params);
        }
        
        public static DSASignature sign(String message, DSAKeyPair keyPair) {
            byte[] hash = sha256(message.getBytes(StandardCharsets.UTF_8));
            BigInteger m = new BigInteger(1, hash);
            
            SecureRandom random = new SecureRandom();
            BigInteger k, r, s;
            
            do {
                do {
                    k = new BigInteger(keyPair.params.q.bitLength() - 1, random);
                    r = keyPair.params.g.modPow(k, keyPair.params.p).mod(keyPair.params.q);
                } while (r.equals(BigInteger.ZERO));
                
                BigInteger kInv = k.modInverse(keyPair.params.q);
                s = kInv.multiply(m.add(keyPair.x.multiply(r))).mod(keyPair.params.q);
            } while (s.equals(BigInteger.ZERO));
            
            return new DSASignature(r, s);
        }
        
        public static boolean verify(String message, DSASignature signature, DSAKeyPair keyPair) {
            byte[] hash = sha256(message.getBytes(StandardCharsets.UTF_8));
            BigInteger m = new BigInteger(1, hash);
            
            if (signature.r.compareTo(BigInteger.ZERO) <= 0 || signature.r.compareTo(keyPair.params.q) >= 0) return false;
            if (signature.s.compareTo(BigInteger.ZERO) <= 0 || signature.s.compareTo(keyPair.params.q) >= 0) return false;
            
            BigInteger w = signature.s.modInverse(keyPair.params.q);
            BigInteger u1 = m.multiply(w).mod(keyPair.params.q);
            BigInteger u2 = signature.r.multiply(w).mod(keyPair.params.q);
            
            BigInteger v = keyPair.params.g.modPow(u1, keyPair.params.p)
                          .multiply(keyPair.y.modPow(u2, keyPair.params.p))
                          .mod(keyPair.params.p)
                          .mod(keyPair.params.q);
            
            return v.equals(signature.r);
        }
    }
    
    /**
     * Diffie-Hellman Key Exchange
     */
    public static class DiffieHellmanKeyExchange {
        
        public static class DHParameters {
            public final BigInteger p, g;
            
            public DHParameters(BigInteger p, BigInteger g) {
                this.p = p; this.g = g;
            }
        }
        
        public static DHParameters generateParameters(int bitLength) {
            SecureRandom random = new SecureRandom();
            
            // Generate safe prime p = 2q + 1
            BigInteger q, p;
            do {
                q = BigInteger.probablePrime(bitLength - 1, random);
                p = q.multiply(BigInteger.valueOf(2)).add(BigInteger.ONE);
            } while (!p.isProbablePrime(50));
            
            // Find generator g
            BigInteger g;
            do {
                g = new BigInteger(bitLength - 1, random);
            } while (g.compareTo(BigInteger.valueOf(2)) < 0 || 
                     g.compareTo(p.subtract(BigInteger.ONE)) >= 0 ||
                     g.modPow(BigInteger.valueOf(2), p).equals(BigInteger.ONE) ||
                     g.modPow(q, p).equals(BigInteger.ONE));
            
            return new DHParameters(p, g);
        }
        
        public static BigInteger generatePrivateKey(DHParameters params) {
            SecureRandom random = new SecureRandom();
            return new BigInteger(params.p.bitLength() - 1, random);
        }
        
        public static BigInteger generatePublicKey(BigInteger privateKey, DHParameters params) {
            return params.g.modPow(privateKey, params.p);
        }
        
        public static BigInteger computeSharedSecret(BigInteger privateKey, BigInteger otherPublicKey, DHParameters params) {
            return otherPublicKey.modPow(privateKey, params.p);
        }
    }
    
    /**
     * Hash-based Message Authentication Code (HMAC)
     */
    public static class HMACAlgorithm {
        
        public static byte[] hmacSHA256(byte[] key, byte[] message) {
            try {
                javax.crypto.Mac mac = javax.crypto.Mac.getInstance("HmacSHA256");
                SecretKeySpec keySpec = new SecretKeySpec(key, "HmacSHA256");
                mac.init(keySpec);
                return mac.doFinal(message);
            } catch (Exception e) {
                throw new RuntimeException("HMAC computation failed", e);
            }
        }
        
        public static boolean verifyHMAC(byte[] key, byte[] message, byte[] expectedHMAC) {
            byte[] computedHMAC = hmacSHA256(key, message);
            return Arrays.equals(computedHMAC, expectedHMAC);
        }
    }
    
    /**
     * Password-Based Key Derivation Function (PBKDF2)
     */
    public static class PBKDF2 {
        
        public static byte[] deriveKey(String password, byte[] salt, int iterations, int keyLength) {
            try {
                javax.crypto.spec.PBEKeySpec spec = new javax.crypto.spec.PBEKeySpec(
                    password.toCharArray(), salt, iterations, keyLength * 8);
                javax.crypto.SecretKeyFactory factory = javax.crypto.SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
                return factory.generateSecret(spec).getEncoded();
            } catch (Exception e) {
                throw new RuntimeException("PBKDF2 key derivation failed", e);
            }
        }
        
        public static byte[] generateSalt() {
            SecureRandom random = new SecureRandom();
            byte[] salt = new byte[16];
            random.nextBytes(salt);
            return salt;
        }
    }
    
    // Utility method for SHA-256 hashing
    private static byte[] sha256(byte[] input) {
        try {
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            return digest.digest(input);
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 hashing failed", e);
        }
    }
    
    private static String bytesToHex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            result.append(String.format("%02x", b));
        }
        return result.toString();
    }
    
    public static void main(String[] args) {
        System.out.println("Cryptography and Security Algorithms Demo:");
        System.out.println("==========================================");
        
        // RSA Demonstration
        System.out.println("1. RSA Cryptography:");
        RSACryptography.RSAKeyPair rsaKeys = RSACryptography.generateKeyPair(1024);
        String message = "Hello, RSA!";
        String encrypted = RSACryptography.encryptString(message, rsaKeys.e, rsaKeys.n);
        String decrypted = RSACryptography.decryptString(encrypted, rsaKeys.d, rsaKeys.n);
        System.out.println("Original: " + message);
        System.out.println("Encrypted: " + encrypted.substring(0, Math.min(50, encrypted.length())) + "...");
        System.out.println("Decrypted: " + decrypted);
        
        // Elliptic Curve Cryptography
        System.out.println("\n2. Elliptic Curve Digital Signature:");
        BigInteger p = new BigInteger("fffffffffffffffffffffffffffffffeffffffffffffffff", 16);
        BigInteger a = new BigInteger("-3");
        BigInteger b = new BigInteger("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b", 16);
        EllipticCurveCryptography.EllipticCurve curve = new EllipticCurveCryptography.EllipticCurve(a, b, p);
        
        BigInteger gx = new BigInteger("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296", 16);
        BigInteger gy = new BigInteger("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5", 16);
        EllipticCurveCryptography.Point basePoint = new EllipticCurveCryptography.Point(gx, gy);
        
        BigInteger order = new BigInteger("fffffffffffffffffffffffffffff99def836146bc9b1b4d22831", 16);
        BigInteger privateKey = new BigInteger("12345678901234567890123456789012", 16);
        EllipticCurveCryptography.Point publicKey = curve.multiply(privateKey, basePoint);
        
        String testMessage = "Test message for ECDSA";
        EllipticCurveCryptography.ECDSASignature signature = 
            EllipticCurveCryptography.sign(testMessage, privateKey, curve, basePoint, order);
        boolean isValid = EllipticCurveCryptography.verify(testMessage, signature, publicKey, curve, basePoint, order);
        
        System.out.println("Message: " + testMessage);
        System.out.println("Signature valid: " + isValid);
        
        // AES Encryption
        System.out.println("\n3. AES Encryption:");
        byte[] aesKey = AESCryptography.generateKey();
        String plaintext = "This is a secret message!";
        byte[] ciphertext = AESCryptography.encrypt(plaintext.getBytes(StandardCharsets.UTF_8), aesKey);
        byte[] decryptedBytes = AESCryptography.decrypt(ciphertext, aesKey);
        String decryptedText = new String(decryptedBytes, StandardCharsets.UTF_8);
        
        System.out.println("Plaintext: " + plaintext);
        System.out.println("Key: " + bytesToHex(aesKey).substring(0, 32) + "...");
        System.out.println("Ciphertext: " + bytesToHex(ciphertext).substring(0, 32) + "...");
        System.out.println("Decrypted: " + decryptedText);
        
        // Diffie-Hellman Key Exchange
        System.out.println("\n4. Diffie-Hellman Key Exchange:");
        DiffieHellmanKeyExchange.DHParameters dhParams = 
            DiffieHellmanKeyExchange.generateParameters(512);
        
        // Alice's keys
        BigInteger alicePrivate = DiffieHellmanKeyExchange.generatePrivateKey(dhParams);
        BigInteger alicePublic = DiffieHellmanKeyExchange.generatePublicKey(alicePrivate, dhParams);
        
        // Bob's keys
        BigInteger bobPrivate = DiffieHellmanKeyExchange.generatePrivateKey(dhParams);
        BigInteger bobPublic = DiffieHellmanKeyExchange.generatePublicKey(bobPrivate, dhParams);
        
        // Shared secrets
        BigInteger aliceShared = DiffieHellmanKeyExchange.computeSharedSecret(alicePrivate, bobPublic, dhParams);
        BigInteger bobShared = DiffieHellmanKeyExchange.computeSharedSecret(bobPrivate, alicePublic, dhParams);
        
        System.out.println("Alice's public key: " + alicePublic.toString(16).substring(0, 20) + "...");
        System.out.println("Bob's public key: " + bobPublic.toString(16).substring(0, 20) + "...");
        System.out.println("Shared secrets match: " + aliceShared.equals(bobShared));
        
        // HMAC Authentication
        System.out.println("\n5. HMAC Authentication:");
        byte[] hmacKey = "secret_key".getBytes(StandardCharsets.UTF_8);
        String hmacMessage = "Authenticated message";
        byte[] hmac = HMACAlgorithm.hmacSHA256(hmacKey, hmacMessage.getBytes(StandardCharsets.UTF_8));
        boolean hmacValid = HMACAlgorithm.verifyHMAC(hmacKey, hmacMessage.getBytes(StandardCharsets.UTF_8), hmac);
        
        System.out.println("Message: " + hmacMessage);
        System.out.println("HMAC: " + bytesToHex(hmac).substring(0, 20) + "...");
        System.out.println("HMAC valid: " + hmacValid);
        
        // PBKDF2 Key Derivation
        System.out.println("\n6. PBKDF2 Key Derivation:");
        String password = "user_password";
        byte[] salt = PBKDF2.generateSalt();
        byte[] derivedKey = PBKDF2.deriveKey(password, salt, 10000, 32);
        
        System.out.println("Password: " + password);
        System.out.println("Salt: " + bytesToHex(salt));
        System.out.println("Derived key: " + bytesToHex(derivedKey));
        
        System.out.println("\nCryptography demonstration completed!");
    }
}
