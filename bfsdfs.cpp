#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }

    void parallelBFS(int startNode) {
        vector<bool> visited(V, false);
        vector<int> current_level;

        visited[startNode] = true;
        current_level.push_back(startNode);

        cout << "Parallel BFS: ";

        while (!current_level.empty()) {
            vector<int> next_level;

            #pragma omp parallel
            {
                vector<int> local_next;
                
                #pragma omp for nowait
                for (int i = 0; i < current_level.size(); i++) {
                    int u = current_level[i];

                    #pragma omp critical
                    cout << u << " ";

                    for (int v : adj[u]) {
                        bool already_visited;
                        #pragma omp critical
                        {
                            already_visited = visited[v];
                            if (!already_visited)
                                visited[v] = true;
                        }

                        if (!already_visited) {
                            local_next.push_back(v);
                        }
                    }
                }

                #pragma omp critical
                next_level.insert(next_level.end(), local_next.begin(), local_next.end());
            }
            current_level = next_level;
        }
        cout << endl;
    }

    void parallelDFS(int startNode) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS: ";

        #pragma omp parallel
        {
            #pragma omp single
            {
                dfsRecursive(startNode, visited);
            }
        }
        cout << endl;
    }

    void dfsRecursive(int u, vector<bool> &visited) {
        bool should_visit = false;

        #pragma omp critical
        {
            if (!visited[u]) {
                visited[u] = true;
                should_visit = true;
            }
        }

        if (should_visit) {
            #pragma omp critical
            cout << u << " ";

            for (int v : adj[u]) {
                #pragma omp task
                dfsRecursive(v, visited);
            }
        }
    }
};

int main() {
    int nodes = 7;
    Graph g(nodes);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    cout << "Running with " << omp_get_max_threads() << " threads.\n" << endl;

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}


// g++ -fopenmp bfsdfs.cpp -o bfsdfs.exe
// ./bfsdfs.exe
