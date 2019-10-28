#include <stdio.h>

int main()
{
    int a, b, c, d, e, f, g, sum = 0;
    float A[10][10], B[10][10], C[10][10];

    printf("enter the number of rows and columns of the A matrix\n");
    scanf("%d%d", &a, &b);
    printf("enter elements of the A matrix\n");

    for (c = 0; c < a; c++)
        for (d = 0; d < b; d++){
            printf("enter the matrix element A[%d][%d] = ",c,d);
            scanf("%f",&A[c][d]);
        }   
    printf("enter the number of rows and columns of the B matrix\n");
    scanf("%d%d",&e,&f);

    if (b != e)
        printf("the matrices can't be Cplied with each other\n");
    else
    {
        printf("enter the elements of B matrix\n");
        
        for (c = 0; c < e; c++)
            for (d = 0; d < f; d++){
                printf("enter the matrix element B[%d][%d] = ",c,d);
                scanf("%f", &B[c][d]);
            }
        for (c = 0; c < a; c++){
            for (d = 0; d < f; d++){
                for (g = 0; g < e; g++){
                    sum = sum + A[c][g]*B[g][d];
                }
                C[c][d] = sum;
                sum = 0;
                }
        }
        printf("Product of the matrices:\n");
 
        for (c = 0; c < a; c++) {
            for (d = 0; d < f; d++)
                printf("%f\t", C[c][d]);
 
            printf("\n");
        
    }
    
}
}
