#include <stdio.h>
#include <stdlib.h>

struct Node {
    int d;
    struct Node* l;
    struct Node* r;
};

struct Node* new(int d) {
    struct Node* n = malloc(sizeof(struct Node));
    n->d = d;
    n->l = n->r = NULL;
    return n;
}

struct Node* ins(struct Node* r, int d) {
    if (!r) return new(d);
    if (d < r->d)
        r->l = ins(r->l, d);
    else
        r->r = ins(r->r, d);
    return r;
}

struct Node* find(struct Node* r, int d) {
    if (!r || r->d == d) return r;
    if (d < r->d)
        return find(r->l, d);
    return find(r->r, d);
}

void swap(struct Node* r) {
    if (r) {
        struct Node* t = r->l;
        r->l = r->r;
        r->r = t;
        swap(r->l);
        swap(r->r);
    }
}

void in(struct Node* r) {
    if (r) {
        in(r->l);
        printf("%d ", r->d);
        in(r->r);
    }
}

int main() {
    struct Node* r = NULL;
    int c, x;
    while (1) {
        scanf("%d", &c);
        if (!c) break;
        if (c == 1) {
            scanf("%d", &x);
            r = ins(r, x);
        }
        if (c == 2) {
            scanf("%d", &x);
            printf(find(r, x) ? "Y\n" : "N\n");
        }
        if (c == 3) {
            swap(r);
        }
    }
    in(r);
}