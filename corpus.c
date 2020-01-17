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


#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_vector.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include "corpus.h"

/*
 * read corpus from a file
 *
 */

corpus* read_data(const char* data_filename)
{
    FILE *fileptr;
    int label, t_enter, t_exit, length,  count, word, inc, n, nd, nw, corpus_total = 0, min_t = 0, max_t = 0, d;
    corpus* c;
    int ngroups = omp_get_num_procs();
    gsl_rng* r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, (unsigned long) time(NULL));

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    fileptr = fopen(data_filename, "r");
    if (fileptr == NULL) return NULL;
    nd = 0; nw = 0;
    c->docs = malloc(sizeof(doc) * 1);
    if (c->docs != NULL)
    {
        int pos = 0;
        char line[5000]; //fscanf(fileptr, "%d,%d,%d,%d", &label, &t_enter, &t_exit, &length) != EOF)
        char* linefeed = line;
        while (fgets(linefeed, sizeof(char) * 10000, fileptr) != NULL)
        {
            //printf("%s", linefeed);
            if (sscanf(linefeed, "%10d,%10d,%10d,%10d%n", &label, &t_enter, &t_exit, &length, &pos) == 0)
            {
                printf("Did not read expected first 4 fields (label, start date, exit date, number of codes) from row %d \n", nd);
                printf("%10d,%10d,%10d,%10d", label, t_enter, t_exit, length);
                linefeed += pos;
                continue;
            }
            linefeed += pos;
            //printf("Document number:%d read in %d digits ", nd, pos);
            //printf("%d,%d,%d,%d \n", label, t_enter, t_exit, length);
            c->docs = (doc*)realloc(c->docs, sizeof(doc) * ((int) nd + 1));
            if (c->docs != NULL)
            {
                c->docs[nd].label = label;
                c->docs[nd].t_enter = t_enter;
                c->docs[nd].t_exit = t_exit;
                c->docs[nd].nterms = length;
                c->docs[nd].total = 0;
                if (length > 0)
                {
                    c->docs[nd].word = malloc(sizeof(int) * length);
                    if (c->docs[nd].word == NULL)
                    {
                        printf("Out of memory\n");
                        return NULL;
                    }
                    c->docs[nd].count = malloc(sizeof(int) * length);
                    if (c->docs[nd].count == NULL)
                    {
                        printf("Out of memory\n");
                        return NULL;
                    }
                    min_t = ((nd == 0 || min_t > t_enter) ? t_enter : min_t);
                    max_t = ((nd == 0 || max_t < t_exit) ? t_exit : max_t);

                    for (n = 0; n < length; n++)
                    {
                        if (sscanf(linefeed, " ,%10d:%10d%n", &word, &count, &inc) ==0)
                        {
                            printf("Did not read expected duple fields (word, count) from row %d word %d\n", nd, n);
                            printf("%10d,%10d,%10d,%10d", label, t_enter, t_exit, length);
                        }

                        linefeed += inc;
                        word = word - OFFSET;
                        //printf(" ,%d:%d",word,count);
                        c->docs[nd].word[n] = word;
                        c->docs[nd].count[n] = count;
                        c->docs[nd].total += count;
                        if (word >= nw) { nw = word + 1; }
                    }
                }
                //printf("\n");
                corpus_total += c->docs[nd].total;
                nd++;
                linefeed = line;
            }
            else
            {
                free(c->docs);
                printf("Out of memory\n");
                return NULL;
            }
        }
        fclose(fileptr);
        c->ndocs = nd;
        c->nterms = nw;
        c->min_t = min_t;
        c->max_t = max_t;
        for (d = 0; d < nd; d++)
        {
            c->docs[d].t_enter -= min_t;
            c->docs[d].t_exit -= min_t;
        }
    }

    //Allocate each person to a random subset for distributed cox regression
    gsl_vector* random = gsl_vector_calloc(nd);
    gsl_vector* permuted = gsl_vector_calloc(nd);
    for (d = 0; d < nd; d++)
    {
        vset(random, d, gsl_rng_uniform(r));
        vset(permuted, d, d);
    }
    gsl_sort_vector2(random, permuted);
    int group_length = (int) ceil( (double) nd / (double) ngroups);
    c->group = gsl_matrix_calloc(group_length, ngroups);
    gsl_matrix_set_all(c->group,nd);
    gsl_vector* temp = gsl_vector_calloc(group_length);
    int cumulative = 0;
    for (int g = 0; g < ngroups; g++)
    {
        for (d = 0; d < group_length; d++)
        {
            if (cumulative >= nd)
            {
                vset(temp, d, nd);
            }
            else
            {
                vset(temp, d, vget(permuted, cumulative));
                cumulative++;
            }
        }
        gsl_sort_vector(temp);
        gsl_matrix_set_col(c->group, g, temp);
        for (d = 0; d < group_length; d++)
            if (mget(c->group,d,g) >= nd) mset(c->group, d, g, -1);
    }
    gsl_rng_free(r);
    gsl_vector_free(temp);
    gsl_vector_free(random);
    gsl_vector_free(permuted);
    
    //set counts of events for each time point within each group for efficient cox regression
    int events = 0;
    c->mark = gsl_matrix_calloc(group_length, ngroups);
    gsl_matrix_set_zero(c->mark);

    gsl_vector* gevents = gsl_vector_calloc(ngroups);
    gsl_vector_set_zero(gevents);

    for (d = group_length - 1; d >= 0; d--)
    {
        for (int g = 0; g < ngroups; g++)
        {
            int gd = (int) mget(c->group, g, d);
            int gprevd = 0;
            if(d != 0) gprevd = (int) mget(c->group, g, d - 1);
            if (d == 0 || c->docs[gd].t_exit != c->docs[gprevd].t_exit)
            {
                mset(c->mark, d, g, vget(gevents, g) + c->docs[gd].label); //Last patient at this time point - store number of deaths
                vset(gevents, g, 0);
            }
            else if (c->docs[gd].t_exit == c->docs[gprevd].t_exit)
                vinc(gevents, g, c->docs[gd].label);
        }
    }

    count = 0;
    c->cmark = gsl_vector_calloc(nd);
    gsl_vector_set_zero(c->cmark);
    c->zbeta = gsl_vector_calloc(nd);
    gsl_vector_set_zero(c->zbeta);
    for (d = nd - 1; d >= 0; d--)
    {
        if (d == 0 || c->docs[d].t_exit != c->docs[d - 1].t_exit)
        {
            vset(c->cmark, d, (double) count + c->docs[d].label);
            count = 0;
        }
        else if (c->docs[d].t_exit == c->docs[d - 1].t_exit)
            count += c->docs[d].label;
    }

    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    printf("total             : %d\n", corpus_total);
    printf("Earliest entry: %d, and latest exit: %d\n", min_t, max_t);

    gsl_vector_free(gevents);
    return(c);
}


/*
 * print document
 *
 */

void print_doc(doc* d)
{
    int i;
    printf("total : %d\n", d->total);
    printf("nterm : %d\n", d->nterms);
    for (i = 0; i < d->nterms; i++)
    {
        printf("%d:%d ", d->word[i], d->count[i]);
    }
}


/*
 * write a corpus to file
 *
 */

void write_corpus(corpus* c, char* filename)
{
    int i, j;
    FILE * fileptr;
    doc * d;

    fileptr = fopen(filename, "w");
    for (i = 0; i < c->ndocs; i++)
    {
        d = &(c->docs[i]);
        fprintf(fileptr, "%d", d->nterms);
        for (j = 0; j < d->nterms; j++)
        {
            fprintf(fileptr, " %d:%d", d->word[j], d->count[j]);
        }
        fprintf(fileptr, "\n");
    }
}


void init_doc(doc* d, int max_nterms)
{
    int i;
    d->nterms = 0;
    d->total = 0;
    d->word = malloc(sizeof(int) * max_nterms);
    d->count = malloc(sizeof(int) * max_nterms);
    if (d->word != NULL && d->count != NULL && max_nterms > 0)
    {
        for (i = 0; i < max_nterms; i++)
        {
            d->word[i] = 0;
            d->count[i] = 0;
        }
    }
}


/*
 * return the 'n'th word of a document
 * (note order has been lost in the representation)
 *
 */

int remove_word(int n, doc* d)
{
    int i = -1, word, pos = 0;
    do
    {
        i++;
        pos += d->count[i];
        word = d->word[i];
    }
    while (pos <= n);
    d->total--;
    d->count[i]--;
    assert(d->count[i] >= 0);
    return(word);
}


/*
 * randomly move some proportion of words from one document to another
 *
 */

void split(doc* orig, doc* dest, double prop)
{
    int w, i, nwords;

    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(r, (long) seed);

    nwords = (int) floor((double) prop * orig->total);
    if (nwords == 0) nwords = 1;
    init_doc(dest, nwords);
    for (i = 0; i < nwords; i++)
    {
        w = remove_word((int) floor(gsl_rng_uniform(r)*orig->total), orig);
        dest->total++;
        dest->nterms++;
        dest->word[i] = w;
        dest->count[i] = 1;
    }
}
