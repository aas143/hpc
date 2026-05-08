#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void seqBubble(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
}

void parBubble(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2)
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
    }
}

void merge(vector<int>& a, int l, int m, int r) {
    vector<int> t;
    int i = l, j = m + 1;
    while (i <= m && j <= r)
        t.push_back(a[i] < a[j] ? a[i++] : a[j++]);
    while (i <= m) t.push_back(a[i++]);
    while (j <= r) t.push_back(a[j++]);
    for (int k = 0; k < t.size(); k++)
        a[l + k] = t[k];
}

void seqMerge(vector<int>& a, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        seqMerge(a, l, m);
        seqMerge(a, m + 1, r);
        merge(a, l, m, r);
    }
}

void parMerge(vector<int>& a, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            seqMerge(a, l, m);
            #pragma omp section
            seqMerge(a, m + 1, r);
        }
        merge(a, l, m, r);
    }
}

int main() {
    int n = 10000;
    vector<int> arr(n), temp;
    for (int i = 0; i < n; i++) arr[i] = rand() % 10000;

    cout << "Array size: " << n << endl;
    cout << "Threads: " << omp_get_max_threads() << "\n\n";

    double t1, t2;

    temp = arr;
    t1 = omp_get_wtime();
    seqBubble(temp);
    t2 = omp_get_wtime();
    cout << "Sequential Bubble Sort: " << (t2 - t1) * 1000 << " ms" << endl;

    temp = arr;
    t1 = omp_get_wtime();
    parBubble(temp);
    t2 = omp_get_wtime();
    cout << "Parallel Bubble Sort  : " << (t2 - t1) * 1000 << " ms" << endl;

    cout << endl;

    temp = arr;
    t1 = omp_get_wtime();
    seqMerge(temp, 0, n - 1);
    t2 = omp_get_wtime();
    cout << "Sequential Merge Sort : " << (t2 - t1) * 1000 << " ms" << endl;

    temp = arr;
    t1 = omp_get_wtime();
    parMerge(temp, 0, n - 1);
    t2 = omp_get_wtime();
    cout << "Parallel Merge Sort   : " << (t2 - t1) * 1000 << " ms" << endl;

    return 0;
}






// g++ -fopenmp bubblemerge.cpp -o bubblemerge.exe
//  ./bubblemerge.exe  
