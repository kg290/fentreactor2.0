#include <stdio.h>

int s[100], top = -1;

void push(int x) { s[++top] = x; }
int pop() { return s[top--]; }

int main() {
    int n, f = 1;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        push(i);
    while (top > -1)
        f *= pop();
    printf("%d", f);
}