package Algorithms.NetworkAndDistributedSystems;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.net.*;
import java.io.*;

/**
 * Network and Distributed Systems Algorithms
 * Consensus algorithms, network protocols, load balancing, and distributed computing
 */
public class NetworkDistributedAlgorithms {
    
    /**
     * Raft Consensus Algorithm Implementation
     */
    public static class RaftConsensus {
        
        public enum NodeState {
            FOLLOWER, CANDIDATE, LEADER
        }
        
        public static class LogEntry {
            public final int term;
            public final String command;
            public final int index;
            
            public LogEntry(int term, String command, int index) {
                this.term = term;
                this.command = command;
                this.index = index;
            }
            
            public String toString() {
                return String.format("LogEntry[term=%d, index=%d, command='%s']", term, index, command);
            }
        }
        
        public static class RaftNode {
            private final int nodeId;
            private volatile NodeState state;
            private volatile int currentTerm;
            private volatile Integer votedFor;
            private final List<LogEntry> log;
            private volatile int commitIndex;
            private volatile int lastApplied;
            
            // Leader state
            private final Map<Integer, Integer> nextIndex;
            private final Map<Integer, Integer> matchIndex;
            
            // Timing
            private volatile long lastHeartbeat;
            private final Random random;
            private final Set<Integer> otherNodes;
            
            // Election
            private final AtomicInteger votesReceived;
            private volatile boolean electionInProgress;
            
            public RaftNode(int nodeId, Set<Integer> otherNodes) {
                this.nodeId = nodeId;
                this.otherNodes = new HashSet<>(otherNodes);
                this.state = NodeState.FOLLOWER;
                this.currentTerm = 0;
                this.votedFor = null;
                this.log = new ArrayList<>();
                this.commitIndex = -1;
                this.lastApplied = -1;
                this.nextIndex = new HashMap<>();
                this.matchIndex = new HashMap<>();
                this.random = new Random();
                this.votesReceived = new AtomicInteger(0);
                this.lastHeartbeat = System.currentTimeMillis();
                
                // Initialize leader state
                for (int node : otherNodes) {
                    nextIndex.put(node, 0);
                    matchIndex.put(node, -1);
                }
            }
            
            public synchronized void startElection() {
                if (state == NodeState.LEADER) return;
                
                state = NodeState.CANDIDATE;
                currentTerm++;
                votedFor = nodeId;
                votesReceived.set(1); // Vote for self
                electionInProgress = true;
                lastHeartbeat = System.currentTimeMillis();
                
                System.out.printf("Node %d starting election for term %d\n", nodeId, currentTerm);
                
                // Send RequestVote RPCs to all other nodes
                for (int node : otherNodes) {
                    requestVote(node);
                }
                
                // Check if we have majority
                checkElectionResult();
            }
            
            private void requestVote(int candidateId) {
                // Simulate RequestVote RPC
                // In real implementation, this would be a network call
                System.out.printf("Node %d requesting vote from node %d for term %d\n", 
                                 nodeId, candidateId, currentTerm);
            }
            
            public synchronized boolean receiveVoteRequest(int candidateId, int term, int lastLogIndex, int lastLogTerm) {
                if (term > currentTerm) {
                    currentTerm = term;
                    votedFor = null;
                    state = NodeState.FOLLOWER;
                }
                
                if (term < currentTerm) {
                    return false;
                }
                
                if (votedFor == null || votedFor == candidateId) {
                    // Check if candidate's log is at least as up-to-date as ours
                    int ourLastLogTerm = log.isEmpty() ? 0 : log.get(log.size() - 1).term;
                    int ourLastLogIndex = log.size() - 1;
                    
                    boolean logUpToDate = (lastLogTerm > ourLastLogTerm) || 
                                         (lastLogTerm == ourLastLogTerm && lastLogIndex >= ourLastLogIndex);
                    
                    if (logUpToDate) {
                        votedFor = candidateId;
                        lastHeartbeat = System.currentTimeMillis();
                        System.out.printf("Node %d voted for node %d in term %d\n", nodeId, candidateId, term);
                        return true;
                    }
                }
                
                return false;
            }
            
            public synchronized void receiveVoteResponse(int voterNode, boolean voteGranted) {
                if (state != NodeState.CANDIDATE || !electionInProgress) return;
                
                if (voteGranted) {
                    int votes = votesReceived.incrementAndGet();
                    System.out.printf("Node %d received vote from node %d (total: %d)\n", nodeId, voterNode, votes);
                    checkElectionResult();
                }
            }
            
            private void checkElectionResult() {
                int totalNodes = otherNodes.size() + 1;
                int majority = (totalNodes / 2) + 1;
                
                if (votesReceived.get() >= majority) {
                    becomeLeader();
                }
            }
            
            private synchronized void becomeLeader() {
                if (state != NodeState.CANDIDATE) return;
                
                state = NodeState.LEADER;
                electionInProgress = false;
                
                // Initialize leader state
                for (int node : otherNodes) {
                    nextIndex.put(node, log.size());
                    matchIndex.put(node, -1);
                }
                
                System.out.printf("Node %d became leader for term %d\n", nodeId, currentTerm);
                
                // Send initial heartbeats
                sendHeartbeats();
            }
            
            public synchronized void sendHeartbeats() {
                if (state != NodeState.LEADER) return;
                
                for (int node : otherNodes) {
                    sendAppendEntries(node, true);
                }
            }
            
            private void sendAppendEntries(int followerId, boolean isHeartbeat) {
                int prevLogIndex = nextIndex.get(followerId) - 1;
                int prevLogTerm = (prevLogIndex >= 0 && prevLogIndex < log.size()) ? 
                                 log.get(prevLogIndex).term : 0;
                
                List<LogEntry> entries = new ArrayList<>();
                if (!isHeartbeat && nextIndex.get(followerId) < log.size()) {
                    entries = log.subList(nextIndex.get(followerId), log.size());
                }
                
                System.out.printf("Node %d sending %s to node %d (prevLogIndex=%d, entries=%d)\n",
                                 nodeId, isHeartbeat ? "heartbeat" : "append entries", 
                                 followerId, prevLogIndex, entries.size());
            }
            
            public synchronized boolean receiveAppendEntries(int leaderId, int term, int prevLogIndex, 
                                                           int prevLogTerm, List<LogEntry> entries, int leaderCommit) {
                lastHeartbeat = System.currentTimeMillis();
                
                if (term > currentTerm) {
                    currentTerm = term;
                    votedFor = null;
                }
                
                if (term < currentTerm) {
                    return false;
                }
                
                state = NodeState.FOLLOWER;
                
                // Check if log contains entry at prevLogIndex with term prevLogTerm
                if (prevLogIndex >= 0) {
                    if (prevLogIndex >= log.size() || 
                        (prevLogIndex < log.size() && log.get(prevLogIndex).term != prevLogTerm)) {
                        return false;
                    }
                }
                
                // Append new entries
                if (!entries.isEmpty()) {
                    // Remove conflicting entries
                    if (prevLogIndex + 1 < log.size()) {
                        log.subList(prevLogIndex + 1, log.size()).clear();
                    }
                    
                    // Append new entries
                    log.addAll(entries);
                    System.out.printf("Node %d appended %d entries\n", nodeId, entries.size());
                }
                
                // Update commit index
                if (leaderCommit > commitIndex) {
                    commitIndex = Math.min(leaderCommit, log.size() - 1);
                    applyLogEntries();
                }
                
                return true;
            }
            
            public synchronized void addLogEntry(String command) {
                if (state != NodeState.LEADER) {
                    System.out.printf("Node %d is not leader, cannot add entry\n", nodeId);
                    return;
                }
                
                LogEntry entry = new LogEntry(currentTerm, command, log.size());
                log.add(entry);
                System.out.printf("Node %d added log entry: %s\n", nodeId, entry);
                
                // Replicate to followers
                for (int node : otherNodes) {
                    sendAppendEntries(node, false);
                }
            }
            
            private void applyLogEntries() {
                while (lastApplied < commitIndex) {
                    lastApplied++;
                    LogEntry entry = log.get(lastApplied);
                    System.out.printf("Node %d applying command: %s\n", nodeId, entry.command);
                }
            }
            
            public boolean isElectionTimeout() {
                long timeout = 150 + random.nextInt(150); // 150-300ms
                return System.currentTimeMillis() - lastHeartbeat > timeout;
            }
            
            public NodeState getState() { return state; }
            public int getCurrentTerm() { return currentTerm; }
            public int getNodeId() { return nodeId; }
            public List<LogEntry> getLog() { return new ArrayList<>(log); }
        }
    }
    
    /**
     * Byzantine Fault Tolerance (PBFT) Algorithm
     */
    public static class PBFTConsensus {
        
        public enum MessageType {
            REQUEST, PREPARE, COMMIT
        }
        
        public static class PBFTMessage {
            public final MessageType type;
            public final int view;
            public final int sequence;
            public final String operation;
            public final int senderId;
            public final String digest;
            
            public PBFTMessage(MessageType type, int view, int sequence, String operation, int senderId) {
                this.type = type;
                this.view = view;
                this.sequence = sequence;
                this.operation = operation;
                this.senderId = senderId;
                this.digest = computeDigest(operation);
            }
            
            private String computeDigest(String operation) {
                return "digest_" + operation.hashCode();
            }
            
            public String toString() {
                return String.format("PBFTMessage[type=%s, view=%d, seq=%d, sender=%d, op='%s']",
                                   type, view, sequence, senderId, operation);
            }
        }
        
        public static class PBFTNode {
            private final int nodeId;
            private final Set<Integer> otherNodes;
            private volatile int view;
            private volatile int sequence;
            private final Map<String, Set<Integer>> prepareMessages;
            private final Map<String, Set<Integer>> commitMessages;
            private final Queue<PBFTMessage> messageQueue;
            private final Set<String> executedOperations;
            private final int faultTolerance;
            
            public PBFTNode(int nodeId, Set<Integer> otherNodes, int faultTolerance) {
                this.nodeId = nodeId;
                this.otherNodes = new HashSet<>(otherNodes);
                this.view = 0;
                this.sequence = 0;
                this.prepareMessages = new ConcurrentHashMap<>();
                this.commitMessages = new ConcurrentHashMap<>();
                this.messageQueue = new ConcurrentLinkedQueue<>();
                this.executedOperations = ConcurrentHashMap.newKeySet();
                this.faultTolerance = faultTolerance;
            }
            
            public synchronized void processRequest(String operation) {
                if (!isPrimary()) {
                    System.out.printf("Node %d is not primary, forwarding request\n", nodeId);
                    return;
                }
                
                sequence++;
                PBFTMessage request = new PBFTMessage(MessageType.REQUEST, view, sequence, operation, nodeId);
                
                System.out.printf("Primary node %d processing request: %s\n", nodeId, operation);
                
                // Send PREPARE messages to all backup nodes
                for (int node : otherNodes) {
                    sendPrepareMessage(node, request);
                }
                
                // Primary also prepares
                receivePrepareMessage(request);
            }
            
            private boolean isPrimary() {
                return (view % (otherNodes.size() + 1)) == nodeId;
            }
            
            private void sendPrepareMessage(int nodeId, PBFTMessage request) {
                PBFTMessage prepare = new PBFTMessage(MessageType.PREPARE, request.view, 
                                                     request.sequence, request.operation, this.nodeId);
                System.out.printf("Node %d sending PREPARE to node %d: %s\n", this.nodeId, nodeId, prepare);
            }
            
            public synchronized void receivePrepareMessage(PBFTMessage message) {
                String key = message.view + ":" + message.sequence + ":" + message.digest;
                prepareMessages.computeIfAbsent(key, k -> ConcurrentHashMap.newKeySet()).add(message.senderId);
                
                System.out.printf("Node %d received PREPARE from node %d\n", nodeId, message.senderId);
                
                // Check if we have enough PREPARE messages (2f + 1)
                if (prepareMessages.get(key).size() >= 2 * faultTolerance + 1) {
                    sendCommitMessage(message);
                }
            }
            
            private void sendCommitMessage(PBFTMessage originalMessage) {
                PBFTMessage commit = new PBFTMessage(MessageType.COMMIT, originalMessage.view,
                                                   originalMessage.sequence, originalMessage.operation, nodeId);
                
                System.out.printf("Node %d sending COMMIT: %s\n", nodeId, commit);
                
                // Send to all nodes including self
                receiveCommitMessage(commit);
                for (int node : otherNodes) {
                    // In real implementation, send over network
                    System.out.printf("Node %d sending COMMIT to node %d\n", nodeId, node);
                }
            }
            
            public synchronized void receiveCommitMessage(PBFTMessage message) {
                String key = message.view + ":" + message.sequence + ":" + message.digest;
                commitMessages.computeIfAbsent(key, k -> ConcurrentHashMap.newKeySet()).add(message.senderId);
                
                System.out.printf("Node %d received COMMIT from node %d\n", nodeId, message.senderId);
                
                // Check if we have enough COMMIT messages (2f + 1)
                if (commitMessages.get(key).size() >= 2 * faultTolerance + 1) {
                    executeOperation(message);
                }
            }
            
            private void executeOperation(PBFTMessage message) {
                String operationKey = message.view + ":" + message.sequence + ":" + message.operation;
                
                if (!executedOperations.contains(operationKey)) {
                    executedOperations.add(operationKey);
                    System.out.printf("Node %d executing operation: %s (view=%d, seq=%d)\n",
                                     nodeId, message.operation, message.view, message.sequence);
                }
            }
            
            public int getNodeId() { return nodeId; }
            public int getView() { return view; }
            public Set<String> getExecutedOperations() { return new HashSet<>(executedOperations); }
        }
    }
    
    /**
     * Consistent Hashing for Load Balancing
     */
    public static class ConsistentHashing {
        
        public static class ConsistentHashRing {
            private final TreeMap<Long, String> ring;
            private final Map<String, Set<Long>> nodeHashes;
            private final int virtualNodes;
            
            public ConsistentHashRing(int virtualNodes) {
                this.ring = new TreeMap<>();
                this.nodeHashes = new HashMap<>();
                this.virtualNodes = virtualNodes;
            }
            
            public void addNode(String node) {
                Set<Long> hashes = new HashSet<>();
                
                for (int i = 0; i < virtualNodes; i++) {
                    String virtualNode = node + ":" + i;
                    long hash = computeHash(virtualNode);
                    ring.put(hash, node);
                    hashes.add(hash);
                }
                
                nodeHashes.put(node, hashes);
                System.out.printf("Added node %s with %d virtual nodes\n", node, virtualNodes);
            }
            
            public void removeNode(String node) {
                Set<Long> hashes = nodeHashes.get(node);
                if (hashes != null) {
                    for (long hash : hashes) {
                        ring.remove(hash);
                    }
                    nodeHashes.remove(node);
                    System.out.printf("Removed node %s\n", node);
                }
            }
            
            public String getNode(String key) {
                if (ring.isEmpty()) return null;
                
                long hash = computeHash(key);
                Map.Entry<Long, String> entry = ring.ceilingEntry(hash);
                
                if (entry == null) {
                    entry = ring.firstEntry();
                }
                
                return entry.getValue();
            }
            
            public Map<String, Integer> getLoadDistribution(List<String> keys) {
                Map<String, Integer> distribution = new HashMap<>();
                
                for (String key : keys) {
                    String node = getNode(key);
                    distribution.merge(node, 1, Integer::sum);
                }
                
                return distribution;
            }
            
            private long computeHash(String input) {
                // Simple hash function - in practice, use SHA-1 or MD5
                return (long) input.hashCode() & 0x7FFFFFFFL;
            }
            
            public void printRingState() {
                System.out.println("Ring state:");
                for (Map.Entry<Long, String> entry : ring.entrySet()) {
                    System.out.printf("  Hash %d -> Node %s\n", entry.getKey(), entry.getValue());
                }
            }
        }
    }
    
    /**
     * Distributed Hash Table (DHT) Implementation
     */
    public static class DistributedHashTable {
        
        public static class DHTNode {
            private final int nodeId;
            private final Map<String, String> localStorage;
            private final Map<Integer, DHTNode> fingerTable;
            private DHTNode successor;
            private DHTNode predecessor;
            private final int m; // Number of bits in hash space
            
            public DHTNode(int nodeId, int m) {
                this.nodeId = nodeId;
                this.m = m;
                this.localStorage = new ConcurrentHashMap<>();
                this.fingerTable = new ConcurrentHashMap<>();
                this.successor = this;
                this.predecessor = this;
            }
            
            public void join(DHTNode existingNode) {
                if (existingNode != null) {
                    initFingerTable(existingNode);
                    updateOthers();
                    moveKeys();
                } else {
                    // First node in the ring
                    for (int i = 0; i < m; i++) {
                        fingerTable.put(i, this);
                    }
                    successor = this;
                    predecessor = this;
                }
                
                System.out.printf("Node %d joined the DHT\n", nodeId);
            }
            
            private void initFingerTable(DHTNode existingNode) {
                fingerTable.put(0, existingNode.findSuccessor((nodeId + (1 << 0)) % (1 << m)));
                successor = fingerTable.get(0);
                predecessor = successor.predecessor;
                successor.predecessor = this;
                predecessor.successor = this;
                
                for (int i = 1; i < m; i++) {
                    int start = (nodeId + (1 << i)) % (1 << m);
                    if (isBetween(start, nodeId, fingerTable.get(i - 1).nodeId)) {
                        fingerTable.put(i, fingerTable.get(i - 1));
                    } else {
                        fingerTable.put(i, existingNode.findSuccessor(start));
                    }
                }
            }
            
            public DHTNode findSuccessor(int id) {
                if (isBetween(id, nodeId, successor.nodeId) || id == successor.nodeId) {
                    return successor;
                } else {
                    DHTNode node = closestPrecedingFinger(id);
                    if (node == this) {
                        return successor.findSuccessor(id);
                    }
                    return node.findSuccessor(id);
                }
            }
            
            private DHTNode closestPrecedingFinger(int id) {
                for (int i = m - 1; i >= 0; i--) {
                    DHTNode finger = fingerTable.get(i);
                    if (finger != null && isBetween(finger.nodeId, nodeId, id)) {
                        return finger;
                    }
                }
                return this;
            }
            
            private boolean isBetween(int id, int start, int end) {
                if (start < end) {
                    return id > start && id < end;
                } else {
                    return id > start || id < end;
                }
            }
            
            private void updateOthers() {
                for (int i = 0; i < m; i++) {
                    int target = (nodeId - (1 << i) + (1 << m)) % (1 << m);
                    DHTNode p = findPredecessor(target);
                    p.updateFingerTable(this, i);
                }
            }
            
            private DHTNode findPredecessor(int id) {
                DHTNode node = this;
                while (!isBetween(id, node.nodeId, node.successor.nodeId) && id != node.successor.nodeId) {
                    node = node.closestPrecedingFinger(id);
                }
                return node;
            }
            
            private void updateFingerTable(DHTNode s, int i) {
                int start = (nodeId + (1 << i)) % (1 << m);
                if (isBetween(s.nodeId, nodeId, fingerTable.get(i).nodeId)) {
                    fingerTable.put(i, s);
                    DHTNode p = predecessor;
                    if (p != null) {
                        p.updateFingerTable(s, i);
                    }
                }
            }
            
            private void moveKeys() {
                // Move keys that should be stored in this node
                Map<String, String> keysToMove = new HashMap<>();
                
                for (Map.Entry<String, String> entry : successor.localStorage.entrySet()) {
                    int keyHash = computeHash(entry.getKey());
                    if (isBetween(keyHash, predecessor.nodeId, nodeId) || keyHash == nodeId) {
                        keysToMove.put(entry.getKey(), entry.getValue());
                    }
                }
                
                for (String key : keysToMove.keySet()) {
                    localStorage.put(key, keysToMove.get(key));
                    successor.localStorage.remove(key);
                }
                
                System.out.printf("Node %d moved %d keys\n", nodeId, keysToMove.size());
            }
            
            public void put(String key, String value) {
                int keyHash = computeHash(key);
                DHTNode responsible = findSuccessor(keyHash);
                responsible.localStorage.put(key, value);
                System.out.printf("Stored key '%s' at node %d (hash=%d)\n", key, responsible.nodeId, keyHash);
            }
            
            public String get(String key) {
                int keyHash = computeHash(key);
                DHTNode responsible = findSuccessor(keyHash);
                String value = responsible.localStorage.get(key);
                System.out.printf("Retrieved key '%s' from node %d: %s\n", key, responsible.nodeId, value);
                return value;
            }
            
            private int computeHash(String key) {
                return Math.abs(key.hashCode()) % (1 << m);
            }
            
            public void printState() {
                System.out.printf("Node %d: successor=%d, predecessor=%d, keys=%d\n",
                                 nodeId, successor.nodeId, predecessor.nodeId, localStorage.size());
                System.out.printf("  Finger table: ");
                for (int i = 0; i < m; i++) {
                    System.out.printf("%d->%d ", i, fingerTable.get(i).nodeId);
                }
                System.out.println();
            }
        }
    }
    
    /**
     * Network Routing Algorithms
     */
    public static class NetworkRouting {
        
        public static class Router {
            private final String routerId;
            private final Map<String, Integer> routingTable; // destination -> cost
            private final Map<String, String> nextHop; // destination -> next router
            private final Map<String, Router> neighbors;
            
            public Router(String routerId) {
                this.routerId = routerId;
                this.routingTable = new ConcurrentHashMap<>();
                this.nextHop = new ConcurrentHashMap<>();
                this.neighbors = new ConcurrentHashMap<>();
                
                // Distance to self is 0
                routingTable.put(routerId, 0);
                nextHop.put(routerId, routerId);
            }
            
            public void addNeighbor(Router neighbor, int cost) {
                neighbors.put(neighbor.routerId, neighbor);
                routingTable.put(neighbor.routerId, cost);
                nextHop.put(neighbor.routerId, neighbor.routerId);
                
                System.out.printf("Router %s added neighbor %s with cost %d\n", 
                                 routerId, neighbor.routerId, cost);
            }
            
            // Distance Vector Routing (Bellman-Ford)
            public boolean updateRoutingTable() {
                boolean updated = false;
                
                for (Router neighbor : neighbors.values()) {
                    Map<String, Integer> neighborTable = neighbor.getRoutingTable();
                    int linkCost = routingTable.get(neighbor.routerId);
                    
                    for (Map.Entry<String, Integer> entry : neighborTable.entrySet()) {
                        String destination = entry.getKey();
                        int neighborCost = entry.getValue();
                        int newCost = linkCost + neighborCost;
                        
                        if (!routingTable.containsKey(destination) || 
                            newCost < routingTable.get(destination)) {
                            
                            routingTable.put(destination, newCost);
                            nextHop.put(destination, neighbor.routerId);
                            updated = true;
                            
                            System.out.printf("Router %s updated route to %s: cost=%d, next=%s\n",
                                             routerId, destination, newCost, neighbor.routerId);
                        }
                    }
                }
                
                return updated;
            }
            
            // Link State Routing (Dijkstra's Algorithm)
            public void computeShortestPaths(Map<String, Router> allRouters) {
                Set<String> visited = new HashSet<>();
                Map<String, Integer> distances = new HashMap<>();
                Map<String, String> previous = new HashMap<>();
                
                // Initialize distances
                for (String router : allRouters.keySet()) {
                    distances.put(router, Integer.MAX_VALUE);
                }
                distances.put(routerId, 0);
                
                while (visited.size() < allRouters.size()) {
                    // Find unvisited node with minimum distance
                    String current = null;
                    int minDistance = Integer.MAX_VALUE;
                    
                    for (String router : distances.keySet()) {
                        if (!visited.contains(router) && distances.get(router) < minDistance) {
                            minDistance = distances.get(router);
                            current = router;
                        }
                    }
                    
                    if (current == null) break;
                    visited.add(current);
                    
                    // Update distances to neighbors
                    Router currentRouter = allRouters.get(current);
                    for (Map.Entry<String, Router> entry : currentRouter.neighbors.entrySet()) {
                        String neighbor = entry.getKey();
                        int linkCost = currentRouter.routingTable.get(neighbor);
                        int newDistance = distances.get(current) + linkCost;
                        
                        if (newDistance < distances.get(neighbor)) {
                            distances.put(neighbor, newDistance);
                            previous.put(neighbor, current);
                        }
                    }
                }
                
                // Update routing table based on shortest paths
                for (String destination : distances.keySet()) {
                    if (!destination.equals(routerId) && distances.get(destination) != Integer.MAX_VALUE) {
                        routingTable.put(destination, distances.get(destination));
                        
                        // Find next hop
                        String next = destination;
                        while (previous.containsKey(next) && !previous.get(next).equals(routerId)) {
                            next = previous.get(next);
                        }
                        nextHop.put(destination, next);
                    }
                }
            }
            
            public List<String> getPath(String destination) {
                List<String> path = new ArrayList<>();
                String current = routerId;
                path.add(current);
                
                while (!current.equals(destination) && nextHop.containsKey(destination)) {
                    current = nextHop.get(destination);
                    path.add(current);
                    
                    // Avoid infinite loops
                    if (path.size() > 100) break;
                }
                
                return path;
            }
            
            public Map<String, Integer> getRoutingTable() {
                return new HashMap<>(routingTable);
            }
            
            public String getRouterId() { return routerId; }
            
            public void printRoutingTable() {
                System.out.printf("Routing table for %s:\n", routerId);
                for (Map.Entry<String, Integer> entry : routingTable.entrySet()) {
                    String dest = entry.getKey();
                    int cost = entry.getValue();
                    String next = nextHop.get(dest);
                    System.out.printf("  %s: cost=%d, next=%s\n", dest, cost, next);
                }
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Network and Distributed Systems Algorithms Demo:");
        System.out.println("===============================================");
        
        // Raft Consensus demonstration
        System.out.println("1. Raft Consensus Algorithm:");
        Set<Integer> nodes = Set.of(1, 2, 3, 4);
        RaftConsensus.RaftNode node0 = new RaftConsensus.RaftNode(0, nodes);
        
        // Simulate election
        node0.startElection();
        
        // Simulate vote responses
        node0.receiveVoteResponse(1, true);
        node0.receiveVoteResponse(2, true);
        
        // Add log entries
        if (node0.getState() == RaftConsensus.NodeState.LEADER) {
            node0.addLogEntry("CREATE TABLE users");
            node0.addLogEntry("INSERT INTO users VALUES (1, 'Alice')");
        }
        
        // PBFT demonstration
        System.out.println("\n2. PBFT Consensus:");
        Set<Integer> pbftNodes = Set.of(1, 2, 3);
        PBFTConsensus.PBFTNode pbftNode0 = new PBFTConsensus.PBFTNode(0, pbftNodes, 1);
        
        pbftNode0.processRequest("TRANSFER 100 FROM Alice TO Bob");
        
        // Simulate PREPARE messages from other nodes
        PBFTConsensus.PBFTMessage prepare1 = new PBFTConsensus.PBFTMessage(
            PBFTConsensus.MessageType.PREPARE, 0, 1, "TRANSFER 100 FROM Alice TO Bob", 1);
        PBFTConsensus.PBFTMessage prepare2 = new PBFTConsensus.PBFTMessage(
            PBFTConsensus.MessageType.PREPARE, 0, 1, "TRANSFER 100 FROM Alice TO Bob", 2);
        
        pbftNode0.receivePrepareMessage(prepare1);
        pbftNode0.receivePrepareMessage(prepare2);
        
        // Consistent Hashing demonstration
        System.out.println("\n3. Consistent Hashing:");
        ConsistentHashing.ConsistentHashRing hashRing = 
            new ConsistentHashing.ConsistentHashRing(150);
        
        hashRing.addNode("server1");
        hashRing.addNode("server2");
        hashRing.addNode("server3");
        
        List<String> keys = Arrays.asList("user1", "user2", "user3", "user4", "user5", 
                                         "data1", "data2", "data3", "data4", "data5");
        
        System.out.println("Key distribution:");
        for (String key : keys) {
            String node = hashRing.getNode(key);
            System.out.printf("  %s -> %s\n", key, node);
        }
        
        Map<String, Integer> distribution = hashRing.getLoadDistribution(keys);
        System.out.println("Load distribution: " + distribution);
        
        // Add new node and see redistribution
        hashRing.addNode("server4");
        Map<String, Integer> newDistribution = hashRing.getLoadDistribution(keys);
        System.out.println("After adding server4: " + newDistribution);
        
        // DHT demonstration
        System.out.println("\n4. Distributed Hash Table (Chord):");
        DistributedHashTable.DHTNode dht1 = new DistributedHashTable.DHTNode(1, 6);
        DistributedHashTable.DHTNode dht4 = new DistributedHashTable.DHTNode(4, 6);
        DistributedHashTable.DHTNode dht9 = new DistributedHashTable.DHTNode(9, 6);
        
        dht1.join(null); // First node
        dht4.join(dht1);
        dht9.join(dht1);
        
        dht1.printState();
        dht4.printState();
        dht9.printState();
        
        // Store and retrieve data
        dht1.put("key1", "value1");
        dht1.put("key2", "value2");
        dht4.put("key3", "value3");
        
        dht1.get("key1");
        dht4.get("key2");
        dht9.get("key3");
        
        // Network Routing demonstration
        System.out.println("\n5. Network Routing (Distance Vector):");
        NetworkRouting.Router routerA = new NetworkRouting.Router("A");
        NetworkRouting.Router routerB = new NetworkRouting.Router("B");
        NetworkRouting.Router routerC = new NetworkRouting.Router("C");
        NetworkRouting.Router routerD = new NetworkRouting.Router("D");
        
        // Create network topology
        routerA.addNeighbor(routerB, 1);
        routerA.addNeighbor(routerC, 4);
        routerB.addNeighbor(routerA, 1);
        routerB.addNeighbor(routerC, 2);
        routerB.addNeighbor(routerD, 5);
        routerC.addNeighbor(routerA, 4);
        routerC.addNeighbor(routerB, 2);
        routerC.addNeighbor(routerD, 1);
        routerD.addNeighbor(routerB, 5);
        routerD.addNeighbor(routerC, 1);
        
        // Run distance vector algorithm
        List<NetworkRouting.Router> routers = Arrays.asList(routerA, routerB, routerC, routerD);
        for (int i = 0; i < 3; i++) {
            System.out.println("Iteration " + (i + 1) + ":");
            for (NetworkRouting.Router router : routers) {
                router.updateRoutingTable();
            }
        }
        
        routerA.printRoutingTable();
        
        System.out.println("Path from A to D: " + routerA.getPath("D"));
        
        System.out.println("\nNetwork and distributed systems demonstration completed!");
    }
}
