package Algorithms.ComputerScience.GraphAlgorithms;

import java.util.*;

/**
 * Graph Algorithms Implementation
 */
public class GraphAlgorithms {
    
    /**
     * Adjacency List representation of Graph
     */
    static class Graph {
        private int V; // Number of vertices
        private LinkedList<Integer>[] adj; // Adjacency lists
        
        public Graph(int V) {
            this.V = V;
            adj = new LinkedList[V];
            for (int i = 0; i < V; ++i)
                adj[i] = new LinkedList<>();
        }
        
        public void addEdge(int v, int w) {
            adj[v].add(w);
        }
        
        public LinkedList<Integer>[] getAdj() {
            return adj;
        }
        
        public int getV() {
            return V;
        }
    }
    
    /**
     * Depth-First Search (DFS) traversal
     * @param graph The graph
     * @param startVertex Starting vertex
     */
    public static void DFS(Graph graph, int startVertex) {
        boolean[] visited = new boolean[graph.getV()];
        System.out.print("DFS traversal starting from vertex " + startVertex + ": ");
        DFSUtil(graph, startVertex, visited);
        System.out.println();
    }
    
    private static void DFSUtil(Graph graph, int v, boolean[] visited) {
        visited[v] = true;
        System.out.print(v + " ");
        
        for (Integer adj : graph.getAdj()[v]) {
            if (!visited[adj]) {
                DFSUtil(graph, adj, visited);
            }
        }
    }
    
    /**
     * Breadth-First Search (BFS) traversal
     * @param graph The graph
     * @param startVertex Starting vertex
     */
    public static void BFS(Graph graph, int startVertex) {
        boolean[] visited = new boolean[graph.getV()];
        LinkedList<Integer> queue = new LinkedList<>();
        
        visited[startVertex] = true;
        queue.add(startVertex);
        
        System.out.print("BFS traversal starting from vertex " + startVertex + ": ");
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");
            
            for (Integer adj : graph.getAdj()[vertex]) {
                if (!visited[adj]) {
                    visited[adj] = true;
                    queue.add(adj);
                }
            }
        }
        System.out.println();
    }
    
    /**
     * Dijkstra's shortest path algorithm
     * @param graph Weighted graph represented as adjacency matrix
     * @param src Source vertex
     * @return Array of shortest distances from source
     */
    public static int[] dijkstra(int[][] graph, int src) {
        int V = graph.length;
        int[] dist = new int[V];
        boolean[] sptSet = new boolean[V];
        
        // Initialize distances
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance(dist, sptSet);
            sptSet[u] = true;
            
            for (int v = 0; v < V; v++) {
                if (!sptSet[v] && graph[u][v] != 0 && 
                    dist[u] != Integer.MAX_VALUE && 
                    dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }
        
        return dist;
    }
    
    private static int minDistance(int[] dist, boolean[] sptSet) {
        int min = Integer.MAX_VALUE;
        int minIndex = -1;
        
        for (int v = 0; v < dist.length; v++) {
            if (!sptSet[v] && dist[v] <= min) {
                min = dist[v];
                minIndex = v;
            }
        }
        
        return minIndex;
    }
    
    /**
     * Floyd-Warshall algorithm for all-pairs shortest paths
     * @param graph Weighted graph as adjacency matrix
     * @return Matrix of shortest distances between all pairs
     */
    public static int[][] floydWarshall(int[][] graph) {
        int V = graph.length;
        int[][] dist = new int[V][V];
        
        // Initialize distance matrix
        for (int i = 0; i < V; i++) {
            System.arraycopy(graph[i], 0, dist[i], 0, V);
        }
        
        // Add all vertices one by one to the set of intermediate vertices
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != Integer.MAX_VALUE && 
                        dist[k][j] != Integer.MAX_VALUE && 
                        dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        return dist;
    }
    
    /**
     * Topological sorting using DFS
     * @param graph Directed Acyclic Graph
     * @return Topologically sorted vertices
     */
    public static List<Integer> topologicalSort(Graph graph) {
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[graph.getV()];
        
        for (int i = 0; i < graph.getV(); i++) {
            if (!visited[i]) {
                topologicalSortUtil(graph, i, visited, stack);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        while (!stack.isEmpty()) {
            result.add(stack.pop());
        }
        
        return result;
    }
    
    private static void topologicalSortUtil(Graph graph, int v, boolean[] visited, Stack<Integer> stack) {
        visited[v] = true;
        
        for (Integer adj : graph.getAdj()[v]) {
            if (!visited[adj]) {
                topologicalSortUtil(graph, adj, visited, stack);
            }
        }
        
        stack.push(v);
    }
    
    /**
     * Detect cycle in directed graph using DFS
     * @param graph Directed graph
     * @return true if cycle exists, false otherwise
     */
    public static boolean hasCycle(Graph graph) {
        int V = graph.getV();
        boolean[] visited = new boolean[V];
        boolean[] recStack = new boolean[V];
        
        for (int i = 0; i < V; i++) {
            if (hasCycleUtil(graph, i, visited, recStack)) {
                return true;
            }
        }
        
        return false;
    }
    
    private static boolean hasCycleUtil(Graph graph, int v, boolean[] visited, boolean[] recStack) {
        if (recStack[v]) {
            return true;
        }
        
        if (visited[v]) {
            return false;
        }
        
        visited[v] = true;
        recStack[v] = true;
        
        for (Integer adj : graph.getAdj()[v]) {
            if (hasCycleUtil(graph, adj, visited, recStack)) {
                return true;
            }
        }
        
        recStack[v] = false;
        return false;
    }
    
    public static void main(String[] args) {
        System.out.println("Graph Algorithms:");
        System.out.println("=================");
        
        // Create a graph for DFS and BFS
        Graph g = new Graph(4);
        g.addEdge(0, 1);
        g.addEdge(0, 2);
        g.addEdge(1, 2);
        g.addEdge(2, 0);
        g.addEdge(2, 3);
        g.addEdge(3, 3);
        
        // DFS and BFS traversals
        DFS(g, 2);
        BFS(g, 2);
        
        // Dijkstra's algorithm
        int[][] weightedGraph = {
            {0, 4, 0, 0, 0, 0, 0, 8, 0},
            {4, 0, 8, 0, 0, 0, 0, 11, 0},
            {0, 8, 0, 7, 0, 4, 0, 0, 2},
            {0, 0, 7, 0, 9, 14, 0, 0, 0},
            {0, 0, 0, 9, 0, 10, 0, 0, 0},
            {0, 0, 4, 14, 10, 0, 2, 0, 0},
            {0, 0, 0, 0, 0, 2, 0, 1, 6},
            {8, 11, 0, 0, 0, 0, 1, 0, 7},
            {0, 0, 2, 0, 0, 0, 6, 7, 0}
        };
        
        System.out.println("\nDijkstra's shortest paths from vertex 0:");
        int[] distances = dijkstra(weightedGraph, 0);
        for (int i = 0; i < distances.length; i++) {
            System.out.println("Distance to vertex " + i + ": " + distances[i]);
        }
        
        // Topological sorting
        Graph dag = new Graph(6);
        dag.addEdge(5, 2);
        dag.addEdge(5, 0);
        dag.addEdge(4, 0);
        dag.addEdge(4, 1);
        dag.addEdge(2, 3);
        dag.addEdge(3, 1);
        
        System.out.println("\nTopological Sort:");
        List<Integer> topOrder = topologicalSort(dag);
        System.out.println(topOrder);
        
        // Cycle detection
        Graph cyclicGraph = new Graph(4);
        cyclicGraph.addEdge(0, 1);
        cyclicGraph.addEdge(1, 2);
        cyclicGraph.addEdge(2, 3);
        cyclicGraph.addEdge(3, 1); // Creates a cycle
        
        System.out.println("\nCycle detection:");
        System.out.println("Has cycle: " + hasCycle(cyclicGraph));
    }
}
