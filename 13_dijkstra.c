#include <stdio.h>
#define N 5

int main() {
    int g[N][N], v[N] = {0}, d[N], i, j, min, u;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            scanf("%d", &g[i][j]);
    for (i = 0; i < N; i++)
        d[i] = 1e9;
    d[0] = 0;
    for (i = 0; i < N - 1; i++) {
        min = 1e9;
        for (j = 0; j < N; j++)
            if (!v[j] && d[j] < min) {
                min = d[j];
                u = j;
            }
        v[u] = 1;
        for (j = 0; j < N; j++)
            if (g[u][j] && d[u] + g[u][j] < d[j])
                d[j] = d[u] + g[u][j];
    }
    for (i = 0; i < N; i++)
        printf("%d ", d[i]);
}