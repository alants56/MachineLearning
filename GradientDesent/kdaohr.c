#include <stdio.h>

const int TA = 30;

//training data
float K[] = {2,12,4,6,2, 7,3,5,5,9, 9,3,3,1,3, 3,14,7,5,8, 7,18,0,0,4, 6,4,5,8,3};
float D[] = {2,2,2,2,2, 5,9,5,5,7, 1,0,1,0,6, 6,0,2,7,3, 4,3,3,3,4, 4,5,10,5,7};
float A[] = {2,5,7,9,9, 12,9,9,5,7, 7,4,8,4,1, 15,7,7,17,7, 8,5,11,12,9, 10,8,12,9,12};
float S[] = {4.9,12.8,8.2,10.0,8.2, 10.4,5.6,8.4,6.4,7.6, 13.3,7.3,9.4,8.2,6.8, 7.7,12.9,7.9,8.4,8.5, 7.5,13.4,7.1,6.5,8.1, 8.9,6.4,7.9,9.7,6.9};
float O[] = {8.0,40.7,18.2,15.5,17.6, 30.6,20.5,15.8,15.9,17.1, 32.7,12.6,20.7,14.8,19.1, 19.2,30.6,15.8,19.1,15.4, 17.7,34.8,20.2,12.7,14.6, 23.9,15.2,14.9,27.5,18.5};
float H[] = {12.8,25.1,16.1,27.5,18.6, 16.6,17.3,29.7,20.0,16.5, 20.7,12.9,14.4,21.8,30.2, 20.5,15.5,16.8,28.7,18.5, 17.4,19.5,17.4,15.0,30.7, 12.7,14.5,40.0,18.7,14.1};
float M[] = {17.6,23.7,17.5,17.9,23.3, 19.1,16.7,21.6,20.5,22.1, 24.2,21.9,19.0,19.7,15.3, 16.4,26.5,21.3,15.6,20.3, 22.1,28.8,16.1,15.5,17.5, 22.4,17.9,16.8,23.1,19.7};
float P[] = {15.4,65.4,42.3,57.7,42.3, 65.5,41.4,48.3,34.5,55.2, 84.2,36.8,57.9,26.3,21.1, 48.6,56.8,37.8,59.5,40.5, 51.7,79.3,37.9,41.4,44.8, 61.5,46.2,65.4,65.4,57.7};

float gradienta(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * K[i] * tmp;
    }
    return sum;
}
float gradientb(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * D[i] * tmp;
    }
    return sum;
}
float gradientc(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * A[i] * tmp;
    }
    return sum;
}

float gradientd(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * O[i] * tmp;
    }
    return sum;
}
float gradiente(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * H[i] * tmp;
    }
    return sum;
}
float gradientf(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * M[i] * tmp;
    }
    return sum;
}

float gradientg(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * P[i] * tmp;
    }
    return sum;
}

float gradients(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp = S[i] - a*K[i] - b*D[i] - c*A[i] - d*O[i] - e*H[i] - f*M[i] - g*P[i] - s;
        sum += -2 * tmp;
    }
    return sum;
}

void outputKDAS(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    for ( ; i < TA; ++i) {
        float tmp =  a*K[i] + b*D[i] + c*A[i] + d*O[i] + e*H[i] + f*M[i] + g*P[i] + s;
        printf("%4.1f: %4.0f, %4.0f, %4.0f, %4.1f\n", tmp, K[i], D[i], A[i], S[i]);
    }
}


float loss(float a, float b, float c,float d, float e, float f, float g, float s)
{
    int i = 0;
    float sum = 0;
    for ( ; i < TA; ++i) {
        float tmp =  a*K[i] + b*D[i] + c*A[i] + d*O[i] + e*H[i] + f*M[i] + g*P[i]  + s - S[i];
        sum += (tmp * tmp);
    }
    return sum;
}


float step1 = 0.000001;
float step2 = 0.000001;
float step3 = 0.000001;

int main()
{

    float a = 1;
    float b = 1;
    float c = 1;
    float d = 0.01;
    float e = 0.01;
    float f = 0.01;
    float g = 0.01;
    float s = 1;

    float ga = 0;
    float gb = 0;
    float gc = 0;
    float gd = 0;
    float ge = 0;
    float gf = 0;
    float gg = 0;
    float gs = 0;


    int t = 1000000;
    int i = 0;
    for ( ; i < t; ++i) {
        ga = gradienta(a,b,c,d,e,f,g,s);
        gb = gradientb(a,b,c,d,e,f,g,s);
        gc = gradientc(a,b,c,d,e,f,g,s);
        gd = gradientd(a,b,c,d,e,f,g,s);
        ge = gradiente(a,b,c,d,e,f,g,s);
        gf = gradientf(a,b,c,d,e,f,g,s);
        gg = gradientg(a,b,c,d,e,f,g,s);
        gs = gradients(a,b,c,d,e,f,g,s);

        a = a - step1 * ga ;
        b = b - step1 * gb ;
        c = c - step1 * gc ;
        d = d - step2 * gd ;
        e = e - step2 * ge ;
        f = f - step2 * gf ;
        f = g - step2 * gg ;
        s = s - step3 * gs ;
        printf("%d:S= %f+%fK+%fD+%fA+%fO+%fH+%fM+%fP\n",i,s,a,b,c,d,e,f,g);
    };
    printf("result: S= %f+%fK+%fD+%fA+%fO+%fH+%fM+%fP, loss = %f\n",s,a,b,c,d,e,f,g,loss(a,b,c,d,e,f,g,s));
    printf("ga = %f,gb = %f,gc = %f,gd = %f,ge = %f,gf = %f,gg=%f,gs=%f\n",a,b,c,d,e,f,g,s);
    outputKDAS(a,b,c,d,e,f,g,s);

    return 0;
}
