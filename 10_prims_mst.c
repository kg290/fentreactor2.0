#include <stdio.h>
#define N 5

int main() {
    int g[N][N], v[N] = {0}, c = 0, i, j, min, k, s = 0;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            scanf("%d", &g[i][j]);
    v[0] = 1;
    while (c < N - 1) {
        min = 1e9;
        for (i = 0; i < N; i++)
            if (v[i])
                for (j = 0; j < N; j++)
                    if (!v[j] && g[i][j] && g[i][j] < min) {
                        min = g[i][j];
                        k = i;
                        s = j;
                    }
        v[s] = 1;
        printf("%d-%d %d\n", k, s, g[k][s]);
        c++;
    }
    return 0;
}