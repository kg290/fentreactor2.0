#include <stdio.h>

int s[100], top = -1;

void push(int x) { s[++top] = x; }
int pop() { return s[top--]; }

int main() {
    int n, a = 0, b = 1;
    scanf("%d", &n);
    push(a);
    push(b);
    for (int i = 2; i < n; i++) {
        int x = s[top] + s[top - 1];
        push(x);
    }
    for (int i = 0; i < n; i++)
        printf("%d ", s[i]);
}