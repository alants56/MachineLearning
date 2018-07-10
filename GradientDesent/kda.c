#include <stdio.h>

const int TA = 12;

//training data,K/D/A/S(score)

/*
    float K[12] = {  15,   15,   13,   8,   6,   1,   3,    21,   10,   7,   6,   15};
    float D[12] = {   6,    4,    5,   5,   6,   3,   5,     7,    3,   5,   7,    8};
    float A[12] = {  12,   12,   12,   9,   3,   4,   3,     8,   25,  16,   8,   15};
    float S[12] = {11.1, 13.3, 11.5, 9.7, 7.1, 7.5, 7.5,  10.0, 10.6, 8.4, 8.6, 10.9};
*/

/*
float K[] = {   7,    4,    6,   3,    4,     9,   4,     4,    2,   0,   2,    4,  15,   15,   13,   8,   6,   1,   3,    21,   10,   7,   6,   15};
float D[] = {   5,    1,    1,   4,    3,     1,   3,     7,    2,   8,   3,    6,   6,    4,    5,   5,   6,   3,   5,     7,    3,   5,   7,    8};
float A[] = {   9,    9,    5,   11,   6,     3,   7,     1,    4,   3,   4,    1,  12,   12,   12,   9,   3,   4,   3,     8,   25,  16,   8,   15};
float S[] = { 9.3, 11.6, 10.8, 11.5, 10.3, 12.3, 9.2,   6.5,  9.0, 5.4, 9.2,  7.1,  11.1, 13.3, 11.5, 9.7, 7.1, 7.5, 7.5,  10.0, 10.6, 8.4, 8.6, 10.9};
*/


float K[12] = {0.183,0.243,0.196,0.282,0.243,0.292,0.220,0.218,0.280,0.252,0.168,0.221};
float D[12] = {0.219,0.261,0.243,0.275,0.251,0.350,0.225,0.224,0.279,0.228,0.193,0.290};
float A[12] = {0.174,0.177,0.203,0.179,0.199,0.207,0.207,0.205,0.145,0.184,0.201,0.213};
float S[12] = { 11.1, 13.3, 11.5,  9.7,  7.1,  7.5,  7.5, 10.0, 10.6,  8.4,  8.6, 10.9};


float gradienta(float a, float b, float c)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i];
        sum += -2 * K[i] * tmp;
    }
    return sum;
}
float gradientb(float a, float b, float c)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i];
        sum += -2 * D[i] * tmp;
    }
    return sum;
}
float gradientc(float a, float b, float c)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i];
        sum += -2 * A[i] * tmp;
    }
    return sum;
}

void outputKDAS(float a, float b, float c)
{
    int i = 0;
    for ( ; i < TA; ++i) {
        float tmp =  a*K[i] + b*D[i] + c*A[i];
        printf("%4.1f: %4.0f, %4.0f, %4.0f, %4.1f\n", tmp, K[i], D[i], A[i], S[i]);
    }
}

float step = 0.0001;

float loss(float a, float b, float c)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i];
        sum += (tmp * tmp);
    }
    return sum;
}



int main()
{
    float a = 1.57;
    float b = 16.0;
    float c = 26.94;
    float ga = 0;
    float gb = 0;
    float gc = 0;

    int t = 100000;
    int i = 0;
    for ( ; i < t; ++i) {

        ga = gradienta(a, b, c);
        gb = gradientb(a, b, c);
        gc = gradientc(a, b, c);
        if ((ga <= 0.0000001 && ga >= -0.0000001)
                || (gb <= 0.0000001 && gb >= -0.0000001)
                || (gc <= 0.0000001 && gc >= -0.0000001) )
            break;
        a = a - step * ga;
        b = b - step * gb;
        c = c - step * gc;

        printf("%d: s = %fK + %fD + %fA\n", i, a, b, c);
    }
    printf("result : s = %fK + %fD + %fA; loss = %f\n", a, b, c, loss(a,b,c));
    outputKDAS(a, b, c);

    return 0;
}
