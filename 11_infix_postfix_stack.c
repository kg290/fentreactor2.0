#include <stdio.h>
#include <ctype.h>

char s[100];
int top = -1;

void push(char c) { s[++top] = c; }
char pop() { return s[top--]; }
int prec(char c) {
    if (c == '^') return 3;
    if (c == '*' || c == '/') return 2;
    if (c == '+' || c == '-') return 1;
    return 0;
}

void infixToPostfix(char* exp) {
    char* e = exp;
    while (*e) {
        if (isalnum(*e))
            printf("%c", *e);
        else if (*e == '(')
            push(*e);
        else if (*e == ')') {
            while (s[top] != '(')
                printf("%c", pop());
            pop();
        } else {
            while (top != -1 && prec(s[top]) >= prec(*e))
                printf("%c", pop());
            push(*e);
        }
        e++;
    }
    while (top != -1)
        printf("%c", pop());
}

int main() {
    char exp[100];
    scanf("%s", exp);
    infixToPostfix(exp);
}