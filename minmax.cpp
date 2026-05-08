#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <climits>

using namespace std;

void min_reduction(const vector<int>& arr) {
    int min_val = INT_MAX;
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    cout << "Minimum value: " << min_val << endl;
}

void max_reduction(const vector<int>& arr) {
    int max_val = INT_MIN;
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    cout << "Maximum value: " << max_val << endl;
}

void sum_reduction(const vector<int>& arr) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    cout << "Sum: " << sum << endl;
    cout << "Average: " << (double)sum / arr.size() << endl;
}

int main() {
    int N = 1000000;
    vector<int> arr(N);

    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 10000;
    }

    cout << "Array size: " << N << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);

    return 0;
}

// g++ -fopenmp minmax.cpp -o minmax.exe
// ./minmax.exe
