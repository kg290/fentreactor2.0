#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

struct Node* insert(struct Node* head, int data) {
    struct Node* newNode = malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = head;
    return newNode;
}

struct Node* insertionSort(struct Node* head) {
    struct Node* sorted = NULL;
    struct Node* curr = head;
    while (curr) {
        struct Node** p = &sorted;
        while (*p && (*p)->data < curr->data)
            p = &(*p)->next;
        struct Node* next = curr->next;
        curr->next = *p;
        *p = curr;
        curr = next;
    }
    return sorted;
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
    head = insertionSort(head);
    print(head);
    return 0;
}