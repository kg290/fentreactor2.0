#include <stdio.h>

struct S {
    char n[20];
    int r, t;
};

void swap(struct S* a, struct S* b) {
    struct S t = *a;
    *a = *b;
    *b = t;
}

void insertion(struct S s[], int n, int* sw) {
    for (int i = 1; i < n; i++) {
        int j = i;
        while (j > 0 && s[j].r < s[j - 1].r) {
            swap(&s[j], &s[j - 1]);
            (*sw)++;
            j--;
        }
    }
}

void bubble(struct S s[], int n, int* sw) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (s[j].r > s[j + 1].r) {
                swap(&s[j], &s[j + 1]);
                (*sw)++;
            }
}

void merge(struct S s[], int l, int m, int r, int* sw) {
    int n1 = m - l + 1, n2 = r - m, i, j, k;
    struct S L[100], R[100];
    for (i = 0; i < n1; i++) L[i] = s[l + i];
    for (j = 0; j < n2; j++) R[j] = s[m + 1 + j];
    i = 0; j = 0; k = l;
    while (i < n1 && j < n2)
        s[k++] = (L[i].r < R[j].r) ? L[i++] : R[j++];
    while (i < n1) s[k++] = L[i++];
    while (j < n2) s[k++] = R[j++];
}

void mergeSort(struct S s[], int l, int r, int* sw) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(s, l, m, sw);
        mergeSort(s, m + 1, r, sw);
        merge(s, l, m, r, sw);
    }
}

int main() {
    struct S s[100];
    int n, c, sw = 0;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
        scanf("%s%d%d", s[i].n, &s[i].r, &s[i].t);
    scanf("%d", &c);
    if (c == 1)
        insertion(s, n, &sw);
    if (c == 2)
        bubble(s, n, &sw);
    if (c == 3)
        mergeSort(s, 0, n - 1, &sw);
    for (int i = 0; i < n; i++)
        printf("%s %d %d\n", s[i].n, s[i].r, s[i].t);
    printf("%d", sw);
}