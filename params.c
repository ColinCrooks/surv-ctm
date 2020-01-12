// (C) Copyright 2007, David M. Blei and John D. Lafferty

// This file is part of CTM-C.

// CTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// CTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "params.h"

llna_params PARAMS;

void read_params(char* filename)
{
    FILE* fileptr;
    char string[100];
    int chk=0;
    fileptr = fopen(filename, "r");
    chk+=fscanf(fileptr, "em max iter %d\n", &(PARAMS.em_max_iter));
    chk+=fscanf(fileptr, "var max iter %d\n", &(PARAMS.var_max_iter));
    chk+=fscanf(fileptr, "cg max iter %d\n", &(PARAMS.cg_max_iter));
    chk+=fscanf(fileptr, "surv max iter %d\n", &(PARAMS.surv_max_iter));
    chk+=fscanf(fileptr, "em convergence %lf\n", &(PARAMS.em_convergence));
    chk+=fscanf(fileptr, "var convergence %lf\n", &(PARAMS.var_convergence));
    chk+=fscanf(fileptr, "cg convergence %lf\n", &(PARAMS.cg_convergence));
    chk+=fscanf(fileptr, "surv convergence %lf\n", &(PARAMS.surv_convergence));
    chk+=fscanf(fileptr, "surv penalty %lf\n", &(PARAMS.surv_penalty));
    chk+=fscanf(fileptr, "lag %d\n", &(PARAMS.lag));
    chk+=fscanf(fileptr, "run in %d\n", &(PARAMS.runin));
    chk+=fscanf(fileptr, "covariance estimate %s\n", string);
    if (strcmp(string, "shrinkage") == 0)
        PARAMS.cov_estimate = SHRINK;
    if (strcmp(string, "mle")==0)
        PARAMS.cov_estimate = MLE;
    if (chk < 12) printf("Only read in %d parameters and expected 12", chk);
}


void print_params()
{
    printf("em max iter %d\n", PARAMS.em_max_iter);
    printf("var max iter %d\n", PARAMS.var_max_iter);
    printf("cg max iter %d\n", PARAMS.cg_max_iter);
    printf("surv max iter %d\n", PARAMS.surv_max_iter);
    printf("em convergence %lf\n", PARAMS.em_convergence);
    printf("var convergence %lf\n", PARAMS.var_convergence);
    printf("cg convergence %lf\n", PARAMS.cg_convergence);
    printf("surv convergence %lf\n", PARAMS.surv_convergence);
    printf("penalty convergence %lf\n", PARAMS.surv_penalty);
    printf("lag %d\n", PARAMS.lag);
    printf("run in %d\n", PARAMS.runin);
    printf("shrinkage? %d\n", PARAMS.cov_estimate);
}


void default_params()
{
    PARAMS.em_max_iter = 1000;
    PARAMS.var_max_iter = 500;
    PARAMS.cg_max_iter = 500;
    PARAMS.surv_max_iter = 100;
    PARAMS.em_convergence = 1e-3;
    PARAMS.var_convergence = 1e-5;
    PARAMS.cg_convergence = 1e-5;
    PARAMS.surv_convergence = 1e-5;
    PARAMS.surv_penalty = 1e-2;
    PARAMS.cov_estimate = MLE;
    PARAMS.lag = 1;
}
