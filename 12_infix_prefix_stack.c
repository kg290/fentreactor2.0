#include <stdio.h>
#include <string.h>
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

void reverse(char* exp) {
    int l = strlen(exp);
    for (int i = 0; i < l / 2; i++) {
        char t = exp[i];
        exp[i] = exp[l - i - 1];
        exp[l - i - 1] = t;
    }
}

void infixToPrefix(char* exp) {
    reverse(exp);
    for (int i = 0; exp[i]; i++) {
        if (exp[i] == '(')
            exp[i] = ')';
        else if (exp[i] == ')')
            exp[i] = '(';
    }
    char res[100], *e = exp;
    int j = 0;
    while (*e) {
        if (isalnum(*e))
            res[j++] = *e;
        else if (*e == '(')
            push(*e);
        else if (*e == ')') {
            while (s[top] != '(')
                res[j++] = pop();
            pop();
        } else {
            while (top != -1 && prec(s[top]) > prec(*e))
                res[j++] = pop();
            push(*e);
        }
        e++;
    }
    while (top != -1)
        res[j++] = pop();
    res[j] = '\0';
    reverse(res);
    printf("%s", res);
}

int main() {
    char exp[100];
    scanf("%s", exp);
    infixToPrefix(exp);
}