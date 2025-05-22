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

void preorder(struct Node* r) {
    struct Node* s[100];
    int top = 0;
    if (!r) return;
    s[top++] = r;
    while (top) {
        struct Node* n = s[--top];
        printf("%d ", n->d);
        if (n->r) s[top++] = n->r;
        if (n->l) s[top++] = n->l;
    }
}

void postorder(struct Node* r) {
    struct Node* s1[100], *s2[100];
    int t1 = 0, t2 = 0;
    if (!r) return;
    s1[t1++] = r;
    while (t1) {
        struct Node* n = s1[--t1];
        s2[t2++] = n;
        if (n->l) s1[t1++] = n->l;
        if (n->r) s1[t1++] = n->r;
    }
    while (t2)
        printf("%d ", s2[--t2]->d);
}

int count(struct Node* r) {
    if (!r) return 0;
    return 1 + count(r->l) + count(r->r);
}

void leaf(struct Node* r) {
    if (r) {
        if (!r->l && !r->r)
            printf("%d ", r->d);
        leaf(r->l);
        leaf(r->r);
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
        if (c == 2)
            preorder(r);
        if (c == 3)
            postorder(r);
        if (c == 4)
            printf("%d\n", count(r));
        if (c == 5)
            leaf(r);
    }
}