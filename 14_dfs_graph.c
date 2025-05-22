#include <stdio.h>

int g[10][10], v[10], n;

void dfs(int u) {
    v[u] = 1;
    printf("%d ", u);
    for (int i = 0; i < n; i++)
        if (g[u][i] && !v[i])
            dfs(i);
}

int main() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &g[i][j]);
    dfs(0);
}