#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

void srandom (unsigned seed);

double dboard (int darts);

#define DARTS 10000000
#define ROUNDS 1
#define LEADER 0

int main (int argc, char *argv[])
{

    double homepi, pisum, pi, avepi;
    int taskid, numtasks, rc, i;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    printf("MPI task %d has started ... \n", taskid);

    /* Set seed to task ID */
    srandom (taskid);
    avepi = 0;

    for (i=0; i<ROUNDS; i++)
    {
        homepi = dboard(DARTS);

        /* USE MPI_Reduce to sum values of homepi across all tasks */
        rc = MPI_Reduce(&homepi, &pisum, 1, MPI_DOUBLE, MPI_SUM, LEADER, MPI_COMM_WORLD);

        if (taskid == LEADER)
        {
            pi = pisum/numtasks;
            avepi = ((avepi * i) + pi) / (i+1);
            printf("After %d throws, average value of pi = %10.8f \n", (DARTS *(i+1)), avepi);

        }
    }

    if (taskid == LEADER)
    {
        printf("\n PI to 13 digits : 3.1415926535897 \n");
    }
    
    MPI_Finalize();

    return 0;

}


double dboard (int darts)
{
    #define sqr(x) ((x) *(x))
    long random(void);
    double x_coord, y_coord, pi, r;
    int score, n;
    unsigned int cconst;    /* must be 4-bytes in size */

    if (sizeof(cconst) != 4)
    {
        printf("Wrong data size for cconst variable in dboard routine ! \n");
        exit(1);
    }

    /* 2 bit shifted to MAX_RAND later used to scale random number between 0 and 1 */
    
    cconst = 2 << (31 - 1);
    score = 0;
    
    for (n=1; n<=darts; n++)
    {
        r = (double) random() / cconst;

        x_coord = (2.0 * r) - 1.0;

        r = (double) random() / cconst;

        y_coord = (2.0 * r) - 1.0;

        if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)
        {
            score++;
        }

    }

    /* calculate pi */
    
    pi = 4.0 * (double) score / (double) darts;
    return (pi);
}

