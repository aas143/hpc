#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> a(n);
    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++)
        cin >> a[i];

    int sum = 0, minVal = a[0], maxVal = a[0];

    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < n; i++) {
        sum += a[i];
        if (a[i] < minVal) minVal = a[i];
        if (a[i] > maxVal) maxVal = a[i];
    }

    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << minVal << endl;
    cout << "Maximum = " << maxVal << endl;
    cout << "Average = " << (double)sum / n << endl;

    return 0;
}
