#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

#include "legendre.c"
#include "limits.c"

int PRINTTEMPS;

void print_state (size_t iter, gsl_multifit_fdfsolver *s, int Pl_order, int PRINTCOEFFS);

int main (int argc, char **argv)
{
	int PRINTCOEFFS;
	int PRINTDATA;
	int PRINTFUNC;
	double T1, T2;

	const gsl_multifit_fdfsolver_type *T;
	gsl_multifit_fdfsolver *s;
	gsl_multifit_function_fdf f;
	gsl_vector_view x;

	int ptsno, Pl_order;
	int status, i, j, iter = 0;
	char *datafile;
	FILE *input;
	double rx, ry;

	double *logT = NULL, *y = NULL;
	struct data d;
	double *x_init;

	if (argc < 6 || argc > 8) {
		printf ("\nUsage: ./legendre input.data Pl_order prinfcoeffs_switch printdata_switch printfunc_switch [Tstart Tend]\n\n");
		exit (-1);
	}
	
	datafile = argv[1];
	Pl_order = atoi (argv[2]);
	PRINTCOEFFS = atoi (argv[3]);
	PRINTDATA = atoi (argv[4]);
	PRINTFUNC = atoi (argv[5]);
	
	if (argc == 8) {
		PRINTTEMPS = 1;
		T1 = atof (argv[6]);
		T2 = atof (argv[7]);
	}

	/********************** This is the data to be fitted *********************/

	i = 0;
	input = fopen (datafile, "r");
	if (!input) {
		printf ("File %s cannot be found, aborting.\n", datafile);
		exit (-1);
	}

	while (!feof (input)) {
		if (fscanf (input, "%lf %lf\n", &rx, &ry) == 2) {
			logT = realloc (logT, (i+1)*sizeof(*logT));
			   y = realloc (   y, (i+1)*sizeof(*y));
			logT[i] = rx;
			y[i] = ry;
			i++;
		}
		else break;
	}

	/**************************************************************************/

	ptsno = i;

	d.n = ptsno;
	d.order = Pl_order;
	d.logT = logT;
	d.y = y;

	f.f = &Pl_f;
	f.df = &Pl_df;
	f.fdf = &Pl_fdf;
	f.n = ptsno;
	f.p = Pl_order;
	f.params = &d;

	x_init = malloc (Pl_order * sizeof (*x_init));
	for (i = 0; i < Pl_order; i++)
		x_init[i] = 1.0;

	x = gsl_vector_view_array (x_init, Pl_order);

	/**************************************************************************/

	T = gsl_multifit_fdfsolver_lmsder;
	s = gsl_multifit_fdfsolver_alloc (T, ptsno, Pl_order);
	gsl_multifit_fdfsolver_set (s, &f, &x.vector);

	/**************************************************************************/

	do {
		iter++;
		status = gsl_multifit_fdfsolver_iterate (s);
		if (status) break;

		status = gsl_multifit_test_delta (s->dx, s->x, 1e-4, 1e-4);
	} while (status == GSL_CONTINUE && iter < 500);

	if (PRINTTEMPS)
		printf ("%8.1lf%8.1lf", T1, T2);

	print_state (iter, s, Pl_order, PRINTCOEFFS);

	/**************************************************************************/

	if (PRINTDATA)
		for (i = 0; i < ptsno; i++)
			printf ("%10.3lf %10.3lf %10.3lf\n", logT[i], y[i], y[i]+gsl_vector_get (s->f, i));

	if (PRINTFUNC) {
		double *Plpolys = malloc (Pl_order*sizeof(*Plpolys)), logI;
		for (i = 0; i < 1000; i++) {
			gsl_sf_legendre_Pl_array (Pl_order-1, (double)i/1000.0, Plpolys);
			logI = 0.0;
			for (j = 0; j < Pl_order; j++)
				logI += Plpolys[j]*gsl_vector_get (s->x, j);
			printf ("%10.3lf\t%10.3lf\n", (double) i/1000.0, logI);
		}
		free (Plpolys);
	}
	/**************************************************************************/

	gsl_multifit_fdfsolver_free (s);
	free (logT);
	free (y);

	return 0;
}

void print_state (size_t iter, gsl_multifit_fdfsolver *s, int Pl_order, int PRINTCOEFFS)
{
	int i;

	if (PRINTCOEFFS == 0) return;

	for (i = 0; i < Pl_order; i++) {
		if (PRINTTEMPS)
			printf ("%17.9E", gsl_vector_get (s->x, i));
		else
			printf ("%18.9E", gsl_vector_get (s->x, i));
	}
	printf ("\n");
}
