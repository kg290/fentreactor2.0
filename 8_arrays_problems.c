#include <stdio.h>

int main() {
    int a[] = {2, 2, 3, 4}, n = 4, x = 1;
    for (int i = 0; i < n; i++) {
        x ^= a[i];
    }
    printf("%d\n", x);

    printf("%p\n", (void*)a);

    int mat[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} }, i, j, k, sp = -1;
    for (i = 0; i < 3; i++) {
        int min = mat[i][0], col = 0;
        for (j = 1; j < 3; j++)
            if (mat[i][j] < min) {
                min = mat[i][j];
                col = j;
            }
        for (k = 0; k < 3; k++)
            if (mat[k][col] > mat[i][col])
                break;
        if (k == 3) sp = min;
    }
    printf("%d\n", sp);

    int ms[3][3] = { {2, 7, 6}, {9, 5, 1}, {4, 3, 8} }, f = 1, sum = 15;
    for (i = 0; i < 3; i++) {
        int r = 0, c = 0;
        for (j = 0; j < 3; j++) {
            r += ms[i][j];
            c += ms[j][i];
        }
        if (r != sum || c != sum)
            f = 0;
    }
    int d = 0, e = 0;
    for (i = 0; i < 3; i++) {
        d += ms[i][i];
        e += ms[i][2 - i];
    }
    if (d != sum || e != sum)
        f = 0;
    printf("%s\n", f ? "YES" : "NO");

    int sm[3][3] = { {0, 0, 3}, {0, 0, 0}, {0, 4, 0} };
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            if (sm[i][j])
                printf("%d %d %d\n", i, j, sm[i][j]);
    return 0;
}