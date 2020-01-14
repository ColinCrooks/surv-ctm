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

/*************************************************************************
 *
 * llna.c
 *
 * reading, writing, and initializing a logistic normal allocation model
 *
 *************************************************************************/

#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <time.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"

/*
 * create a new empty model
 *
 */

llna_model* new_llna_model(int ntopics, int nterms, int range_t)
{
    llna_model* model = malloc(sizeof(llna_model));
    if (model != NULL)
    {
        model->k = ntopics;
        model->em_convergence = 0.1;
        model->var_convergence = 0.01;
        model->cg_convergence = 0.01;
        model->surv_convergence = 0.01;
        model->iteration = 0;
        model->range_t = range_t;
        model->mu = gsl_vector_calloc(ntopics - 1);
        model->cov = gsl_matrix_calloc(ntopics - 1, ntopics - 1);
        model->inv_cov = gsl_matrix_calloc(ntopics - 1, ntopics - 1);
        model->log_omega = gsl_matrix_calloc(ntopics, nterms);
        model->topic_beta = gsl_vector_calloc(ntopics);
        model->cbasehazard = gsl_vector_calloc(range_t);
        model->basehazard = gsl_vector_calloc(range_t);
        return(model);
    }
    else
        return NULL;
}


/*
 * create and delete sufficient statistics
 *
 */

llna_ss * new_llna_ss(llna_model* model)
{
    llna_ss * ss;
    ss = malloc(sizeof(llna_ss));
    if (ss != NULL)
    {
        ss->mu_ss = gsl_vector_calloc(model->k - 1);
        ss->cov_ss = gsl_matrix_calloc(model->k - 1, model->k - 1);
        ss->omega_ss = gsl_matrix_calloc(model->k, model->log_omega->size2);
        ss->ndata = 0;
        reset_llna_ss(ss);
        return(ss);
    }
    else
        return NULL;
}


void del_llna_ss(llna_ss * ss)
{
    gsl_vector_free(ss->mu_ss);
    gsl_matrix_free(ss->cov_ss);
    gsl_matrix_free(ss->omega_ss);
}


void reset_llna_ss(llna_ss * ss)
{
    gsl_matrix_set_all(ss->omega_ss, 0);
    gsl_matrix_set_all(ss->cov_ss, 0);
    gsl_vector_set_all(ss->mu_ss, 0);
    ss->ndata = 0;
}


void write_ss(llna_ss * ss)
{
    printf_matrix("cov_ss", ss->cov_ss);
    printf_matrix("omega_ss", ss->omega_ss);
    printf_vector("mu_ss", ss->mu_ss);
}
/*
 * initialize a model with zero-mean, diagonal covariance gaussian and
 * topics seeded from the corpus
 *
 */

llna_model* corpus_init(int ntopics, corpus* corpus)
{
    int range_t = 1 + corpus->max_t - corpus->min_t;
    llna_model* model = new_llna_model(ntopics, corpus->nterms, range_t);
    if (model == NULL)
        return NULL;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
    doc* doc;
    int i, k, n, d, t;
    double sum;
    time_t seed;
    time(&seed);
    printf("SEED = %lld\n", seed);
    printf("USING 1115574245\n");
    gsl_rng_set(r, (long) 1115574245);
    // gsl_rng_set(r, (long) seed);
    // gsl_rng_set(r, (long) 432403824);

    // gaussian
    for (i = 0; i < ntopics-1; i++)
    {
        vset(model->mu, i, 0);
        vset(model->topic_beta, i, 0);
        mset(model->cov, i, i, 1.0);
    }
    matrix_inverse(model->cov, model->inv_cov);
    model->log_det_inv_cov = log_det(model->inv_cov);

    // topics
    for (k = 0; k < ntopics; k++)
    {
        sum = 0;
        // seed
        for (i = 0; i < NUM_INIT; i++)
        {
            d = (int) floor(gsl_rng_uniform(r)*corpus->ndocs);
            printf("initialized with document %d\n", d);
            doc = &(corpus->docs[d]);
            for (n = 0; n < doc->nterms; n++)
            {
                minc(model->log_omega, k, doc->word[n], (double) doc->count[n]);
            }
        }
        // smooth
        for (n = 0; n < model->log_omega->size2; n++)
        {
            minc(model->log_omega, k, n, SEED_INIT_SMOOTH + gsl_rng_uniform(r));
            // minc(model->log_omega, k, n, SEED_INIT_SMOOTH);
            sum += mget(model->log_omega, k, n);
        }
        sum = safe_log(sum);
        // normalize
        for (n = 0; n < model->log_omega->size2; n++)
        {
            mset(model->log_omega, k, n,
                 safe_log(mget(model->log_omega, k, n)) - sum);
        }
    }
    for (t = 0; t < range_t; t++)
        vset(model->cbasehazard, t, 0.0);

    gsl_rng_free(r);
    return(model);
}

/*
 * random initialization means zero-mean, diagonal covariance gaussian
 * and randomly generated topics
 *
 */

llna_model* random_init(int ntopics, int nterms, int range_t)
{
    int i, j, t;
    double sum, val;
    llna_model* model = new_llna_model(ntopics, nterms, range_t);
    if (model == NULL)
    {
        return NULL;
    }
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
  // long t1;
   // (void) time(&t1);
    // !!! DEBUG
   // gsl_rng_set(r, (long) 1115574245);
   // printf("RANDOM SEED = %ld\n", t1);
    gsl_rng_set(r, (unsigned long) time(NULL));


    for (i = 0; i < ntopics-1; i++)
    {
        vset(model->mu, i, 0);
        vset(model->topic_beta, i, 0);
        mset(model->cov, i, i, 1.0);
    }
    for (i = 0; i < ntopics; i++)
    {
        sum = 0;
        for (j = 0; j < nterms; j++)
        {
            val = gsl_rng_uniform(r) + 1.0/100;
            sum += val;
            mset(model->log_omega, i, j, val);
        }
        for (j = 0; j < nterms; j++)
            mset(model->log_omega, i, j, log(mget(model->log_omega, i, j) / sum));
    }
    matrix_inverse(model->cov, model->inv_cov);
    model->log_det_inv_cov = log_det(model->inv_cov);
    for (t = 0; t < range_t; t++) 
        vset(model->cbasehazard, t, 0.0);

    gsl_rng_free(r);
    return(model);
}

/*
 * read a model
 *
 */

llna_model* read_llna_model(char * root)
{
    char filename[200];
    FILE* fileptr;
    llna_model* model;
    int ntopics, nterms, range_t;

    // read parameters
    sprintf(filename, "%s-param.txt", root);
    printf("reading params from %s\n", filename);
    fileptr = fopen(filename, "r");
    if (fscanf(fileptr, "num_topics %d\n", &ntopics) == 0)
    {
        printf("Error reading number of topics\n");
        return NULL;
    }
    if(fscanf(fileptr, "num_terms %d\n", &nterms)==0)
    {
        printf("Error reading number of terms\n");
        return NULL ;
    };
    if(fscanf(fileptr, "Range of times %d\n", &range_t)==0)
    {
        printf("Error reading range of times\n");
        return NULL ;
    };
    fclose(fileptr);
    printf("%d topics, %d terms\n", ntopics, nterms);
    // allocate model
    model = new_llna_model(ntopics, nterms, range_t);
    // read gaussian
    printf("reading gaussian\n");
    sprintf(filename, "%s-mu.dat", root);
    scanf_vector(filename, model->mu);
    sprintf(filename, "%s-cov.dat", root);
    scanf_matrix(filename, model->cov);
    sprintf(filename, "%s-inv-cov.dat", root);
    scanf_matrix(filename, model->inv_cov);
    sprintf(filename, "%s-log-det-inv-cov.dat", root);
    fileptr = fopen(filename, "r");
    if(fscanf(fileptr, "%lf\n", &(model->log_det_inv_cov))==0)
    {
        printf("Error reading log det inv cov\n");
        return NULL;
    };    
    fclose(fileptr);
    // read topic matrix
    printf("reading topics\n");
    sprintf(filename, "%s-log-omega.dat", root);
    scanf_matrix(filename, model->log_omega);
    printf("reading topic coefficients\n");
    sprintf(filename, "%s-topic-beta.dat", root);
    scanf_vector(filename, model->topic_beta);
    printf("reading cumulative baseline hazard\n");
    sprintf(filename, "%s-cum-baseline-hazard.dat", root);
    scanf_vector(filename, model->cbasehazard);
    return(model);
}

/*
 * write a model
 *
 */

void write_llna_model(llna_model * model, char * root)
{
    char filename[200];
    FILE* fileptr;

    // write parameters
    printf("writing params\n");
    sprintf(filename, "%s-param.txt", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "num_topics %d\n", model->k);
    fprintf(fileptr, "num_terms %d\n", (int) model->log_omega->size2);
    fprintf(fileptr, "Range of times %d\n", model->range_t);
    fclose(fileptr);
    // write gaussian
    printf("writing gaussian\n");
    sprintf(filename, "%s-mu.dat", root);
    printf_vector(filename, model->mu);
    sprintf(filename, "%s-cov.dat", root);
    printf_matrix(filename, model->cov);
    sprintf(filename, "%s-inv-cov.dat", root);
    printf_matrix(filename, model->inv_cov);
    sprintf(filename, "%s-log-det-inv-cov.dat", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", model->log_det_inv_cov);
    fclose(fileptr);
    // write topic matrix
    printf("writing topics\n");
    sprintf(filename, "%s-log-omega.dat", root);
    printf_matrix(filename, model->log_omega);
    printf("writing topic coefficients\n");
    sprintf(filename, "%s-topic-beta.dat", root);
    printf_vector(filename, model->topic_beta);
    printf("writing cumulative baseline hazard\n");
    sprintf(filename, "%s-cum-baseline-hazard.dat", root);
    printf_vector(filename, model->cbasehazard);
}
