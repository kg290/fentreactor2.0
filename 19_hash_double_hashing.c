#include <stdio.h>
#define N 10

int h[N];

int h2(int k) {
    return 7 - (k % 7);
}

void insert(int k) {
    int i = k % N, j = 0;
    while (h[i] != -1)
        i = (i + h2(k)) % N;
    h[i] = k;
}

void display() {
    for (int i = 0; i < N; i++)
        printf("%d ", h[i]);
}

int main() {
    for (int i = 0; i < N; i++)
        h[i] = -1;
    int n, x;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &x);
        insert(x);
    }
    display();
}