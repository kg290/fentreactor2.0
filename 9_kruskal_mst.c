#include <stdio.h>
#define N 5

int parent[N];

int find(int x) {
    return parent[x] == x ? x : parent[x] = find(parent[x]);
}

void unite(int x, int y) {
    parent[find(x)] = find(y);
}

int main() {
    int g[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            scanf("%d", &g[i][j]);
    for (int i = 0; i < N; i++)
        parent[i] = i;
    int e = 0;
    while (e < N - 1) {
        int min = 1e9, x = 0, y = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (find(i) != find(j) && g[i][j] && g[i][j] < min) {
                    min = g[i][j];
                    x = i;
                    y = j;
                }
        if (find(x) != find(y)) {
            printf("%d-%d %d\n", x, y, g[x][y]);
            unite(x, y);
            e++;
        }
        g[x][y] = g[y][x] = 0;
    }
    return 0;
}