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
#include <time.h>
#include <assert.h>

#include "corpus.h"

/*
 * read corpus from a file
 *
 */

corpus* read_data(const char* data_filename)
{
    FILE *fileptr;
    int label, t_enter, t_exit, length,  count, word, inc, n, nd, nw, corpus_total = 0, min_t, max_t, d;
    corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    fileptr = fopen(data_filename, "r");
    if (fileptr == NULL) return -1;
    nd = 0; nw = 0;
    c->docs = malloc(sizeof(doc) * 1);
	int pos = 0;
	char* line[5000]; //fscanf(fileptr, "%d,%d,%d,%d", &label, &t_enter, &t_exit, &length) != EOF)
	char* linefeed = line;
	while (fgets(linefeed, sizeof(char) * 5000,fileptr)!=NULL)
    {
	    //printf("%s", linefeed);
	    sscanf(linefeed, "%10d,%10d,%10d,%10d%n", &label, &t_enter, &t_exit, &length,&pos);
	    linefeed += pos;
	    //printf("Document number:%d read in %d digits ", nd, pos);
	    //printf("%d,%d,%d,%d \n", label, t_enter, t_exit, length);
	    c->docs = (doc*) realloc(c->docs, sizeof(doc)*(nd+1));
        if (c->docs == NULL)
        {
            printf("Out of memory\n");
            return;
        }
	    c->docs[nd].label = label;
	    c->docs[nd].t_enter = t_enter;
	    c->docs[nd].t_exit = t_exit;
	    c->docs[nd].nterms = length;
	    c->docs[nd].total = 0;
	    c->docs[nd].word = malloc(sizeof(int)*length);
	    c->docs[nd].count = malloc(sizeof(int)*length);

        min_t = ((nd == 0 || min_t > t_enter ) ? t_enter : min_t);
        max_t = ((nd == 0 || max_t < t_exit) ? t_exit : max_t);

	    for (n = 0; n < length; n++)
	    {
	        sscanf(linefeed, " ,%10d:%10d%n", &word, &count, &inc);
		    linefeed += inc;
	        word = word - OFFSET;
		    //printf(" ,%d:%d",word,count);
	        c->docs[nd].word[n] = word;
	        c->docs[nd].count[n] = count;
	        c->docs[nd].total += count;
	        if (word >= nw) { nw = word + 1; }
	    }
	    //printf("\n");
	    corpus_total += c->docs[nd].total;
        nd++;
	    linefeed = line;
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
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    printf("total             : %d\n", corpus_total);
    printf("Earliest entry: %d, and latest exit: %d\n", min_t, max_t);
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
    for (i = 0; i < max_nterms; i++)
    {
        d->word[i] = 0;
        d->count[i] = 0;
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

    nwords = floor((double) prop * orig->total);
    if (nwords == 0) nwords = 1;
    init_doc(dest, nwords);
    for (i = 0; i < nwords; i++)
    {
        w = remove_word(floor(gsl_rng_uniform(r)*orig->total), orig);
        dest->total++;
        dest->nterms++;
        dest->word[i] = w;
        dest->count[i] = 1;
    }
}
