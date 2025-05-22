#include <stdio.h>
#include <stdlib.h>

struct Node {
    int d;
    struct Node* n;
    struct Node* p;
};

struct Node* push(struct Node* top, int d) {
    struct Node* t = malloc(sizeof(struct Node));
    t->d = d;
    t->n = top;
    t->p = NULL;
    if (top) top->p = t;
    return t;
}

struct Node* pop(struct Node* top) {
    if (!top) return NULL;
    struct Node* t = top;
    top = top->n;
    if (top) top->p = NULL;
    free(t);
    return top;
}

int main() {
    struct Node* top = NULL;
    int c, x;
    while (1) {
        scanf("%d", &c);
        if (!c) break;
        if (c == 1) {
            scanf("%d", &x);
            top = push(top, x);
        }
        if (c == 2)
            top = pop(top);
    }
    while (top) {
        printf("%d ", top->d);
        top = top->n;
    }
}