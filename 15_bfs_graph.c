#include <stdio.h>

int g[10][10], v[10], n;

void bfs(int u) {
    int q[10], f = 0, r = 0;
    q[r++] = u;
    v[u] = 1;
    while (f < r) {
        int x = q[f++];
        printf("%d ", x);
        for (int i = 0; i < n; i++)
            if (g[x][i] && !v[i]) {
                q[r++] = i;
                v[i] = 1;
            }
    }
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &g[i][j]);
    bfs(0);
}