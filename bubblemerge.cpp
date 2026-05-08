#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

// --- BUBBLE SORT ---

void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
        }
    }
}

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) { // Even phase
            #pragma omp parallel for
            for (int j = 0; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
            }
        } else { // Odd phase
            #pragma omp parallel for
            for (int j = 1; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// --- MERGE SORT ---

void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void sequentialMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        // Optimization: only use tasks for large enough subarrays
        if (r - l < 1000) { 
            sequentialMergeSort(arr, l, r);
        } else {
            #pragma omp task shared(arr)
            parallelMergeSort(arr, l, m);
            
            #pragma omp task shared(arr)
            parallelMergeSort(arr, m + 1, r);
            
            #pragma omp taskwait
            merge(arr, l, m, r);
        }
    }
}

// --- UTILS ---

void measurePerformance(string name, vector<int> arr, bool parallel, bool isMerge) {
    auto start = high_resolution_clock::now();
    
    if (isMerge) {
        if (parallel) {
            #pragma omp parallel
            #pragma omp single
            parallelMergeSort(arr, 0, arr.size() - 1);
        } else {
            sequentialMergeSort(arr, 0, arr.size() - 1);
        }
    } else {
        if (parallel) parallelBubbleSort(arr);
        else sequentialBubbleSort(arr);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << name << " took: " << duration.count() << " ms" << endl;
}

int main() {
    int N = 10000; // Size for Bubble Sort (keep small, O(n^2))
    int M = 100000; // Size for Merge Sort (can be larger, O(n log n))

    vector<int> dataB(N), dataM(M);
    for (int i = 0; i < N; i++) dataB[i] = rand() % 10000;
    for (int i = 0; i < M; i++) dataM[i] = rand() % 100000;

    cout << "Threads available: " << omp_get_max_threads() << "\n" << endl;

    cout << "--- Bubble Sort (N=" << N << ") ---" << endl;
    measurePerformance("Sequential Bubble Sort", dataB, false, false);
    measurePerformance("Parallel Bubble Sort  ", dataB, true, false);

    cout << "\n--- Merge Sort (N=" << M << ") ---" << endl;
    measurePerformance("Sequential Merge Sort ", dataM, false, true);
    measurePerformance("Parallel Merge Sort   ", dataM, true, true);

    return 0;
}

// g++ -fopenmp bubblemerge.cpp -o bubblemerge.exe
// ./bubblemerge.exe
