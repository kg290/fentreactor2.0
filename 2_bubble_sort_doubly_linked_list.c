#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* prev;
    struct Node* next;
};

struct Node* insert(struct Node* head, int data) {
    struct Node* n = malloc(sizeof(struct Node));
    n->data = data;
    n->next = head;
    n->prev = NULL;
    if (head)
        head->prev = n;
    return n;
}

void bubbleSort(struct Node* head) {
    int swapped;
    struct Node* p;
    if (!head)
        return;
    do {
        swapped = 0;
        p = head;
        while (p->next) {
            if (p->data > p->next->data) {
                int t = p->data;
                p->data = p->next->data;
                p->next->data = t;
                swapped = 1;
            }
            p = p->next;
        }
    } while (swapped);
}

void print(struct Node* head) {
    while (head) {
        printf("%d ", head->data);
        head = head->next;
    }
}

int main() {
    struct Node* head = NULL;
    int n, x;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &x);
        head = insert(head, x);
    }
    bubbleSort(head);
    print(head);
    return 0;
}