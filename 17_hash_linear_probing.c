#include <stdio.h>
#define N 10

int h[N];

void insert(int k) {
    int i = k % N;
    while (h[i] != -1)
        i = (i + 1) % N;
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