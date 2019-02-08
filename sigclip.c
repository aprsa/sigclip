#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_multifit_nlin.h>
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int Pl_order = 2;

struct data {
	int n;
	int order;
	double *x;
	double *y;
	double *w;
};

int Pl_f (const gsl_vector *x, void *data, gsl_vector *f)
{
	size_t      n = ((struct data *) data)->n;
	double     *X = ((struct data *) data)->x;
	double     *Y = ((struct data *) data)->y;
	int     order = ((struct data *) data)->order;

	double *Plcoeffs, *Plpolys, Plval;
	int i, j;

	Plcoeffs = malloc (order * sizeof(*Plcoeffs));
	Plpolys  = malloc (order * sizeof(*Plpolys));

	/* Read out Legendre coefficients: */
	for (i = 0; i < order; i++)
		Plcoeffs[i] = gsl_vector_get (x, i);

	/* Compute the value of Legendre series: */
	for (i = 0; i < n; i++) {
		gsl_sf_legendre_Pl_array (order-1, X[i], Plpolys);
		Plval = 0.0;
		for (j = 0; j < order; j++)
			Plval += Plcoeffs[j] * Plpolys[j];
		gsl_vector_set (f, i, Plval - Y[i]);
	}

	free (Plcoeffs);
	free (Plpolys);

	return GSL_SUCCESS;
}

int Pl_df (const gsl_vector *x, void *data, gsl_matrix *J)
{
	size_t     n = ((struct data *) data)->n;
	double    *X = ((struct data *) data)->x;
	int     order = ((struct data *) data)->order;

	double *Plcoeffs, *Plpolys;
	int i, j;

	Plcoeffs = malloc (order * sizeof(*Plcoeffs));
	Plpolys  = malloc (order * sizeof(*Plpolys));

	/* Read out Legendre coefficients: */
	for (i = 0; i < order; i++)
		Plcoeffs[i] = gsl_vector_get (x, i);

	for (i = 0; i < n; i++) {
		gsl_sf_legendre_Pl_array (order-1, X[i], Plpolys);
		for (j = 0; j < order; j++)
			gsl_matrix_set (J, i, j, Plpolys[j]);
	}

	free (Plcoeffs);
	free (Plpolys);

	return GSL_SUCCESS;
}

int Pl_fdf (const gsl_vector *x, void *data, gsl_vector *f, gsl_matrix *J)
{
	Pl_f (x, data, f);
	Pl_df (x, data, J);

	return GSL_SUCCESS;
}

double stdev (double *x, int n)
{
	double mean, stdev;
	int i;

	mean = 0.0;
	for (i = 0; i < n; i++)
		mean += x[i];
	mean /= n;
	stdev = 0.0;
	for (i = 0; i < n; i++)
		stdev += (x[i]-mean)*(x[i]-mean);
	return sqrt(stdev/(n-1));
}

int cull (double **x, double **y, double **w, int n, gsl_vector *s, double sig_lo, double sig_hi)
{
	double *nx = NULL, *ny = NULL, *nw = NULL, *f;
	int i, j, nn = 0;

	double *Plpolys = malloc (Pl_order*sizeof(*Plpolys));

	f = malloc (n*sizeof(*f));
	for (i = 0; i < n; i++) {
		gsl_sf_legendre_Pl_array (Pl_order-1, (*x)[i], Plpolys);
		f[i] = 0.0;
		for (j = 0; j < Pl_order; j++)
			f[i] += Plpolys[j]*gsl_vector_get (s, j);
	}
	free (Plpolys);
	
	for (i = 0; i < n; i++) {
		if ((*y)[i]-f[i] < sig_hi && (*y)[i]-f[i] > sig_lo) {
			nn++;
			nx = realloc (nx, nn*sizeof(*nx));
			ny = realloc (ny, nn*sizeof(*ny));
			nw = realloc (nw, nn*sizeof(*nw));
			nx[nn-1] = (*x)[i]; ny[nn-1] = (*y)[i]; nw[nn-1] = (*w)[i];
		}
	}
	
	if (nn > Pl_order) {
		free (f);
		free (*x);
		free (*y);
		free (*w);
	
		*x = nx; *y = ny; *w = nw;
		return nn;
	}
	
	return n;
}

int kill_cr (double **y, int n, double sig)
{
	int i, ncr = 0;

	for (i = 1; i < n-1; i++)
		if ( fabs((*y)[i]-(*y)[i-1]) > sig &&
                     fabs((*y)[i+1]-(*y)[i]) > sig &&
                     fabs((*y)[i+1]-(*y)[i-1]) < sig) {
			(*y)[i] = 0.5*((*y)[i-1]+(*y)[i+1]);
			ncr++;
		}
	
	return ncr;
}

int sorter (const void *a, const void *b)
{
	const double *da = (const double *) a;
	const double *db = (const double *) b;
	
	return (da > db) - (da < db);
}

double median (double *y, int n)
{
	int i;
	double *sy = malloc(n*sizeof(*sy));
	double retval;
	
	for (i = 0; i < n; i++)
		sy[i] = y[i];
	qsort(sy, n, sizeof(*sy), sorter);

	if (n % 2 == 0)
		retval = 0.5*(sy[n/2-1]+sy[n/2]);
	else
		retval = sy[n/2-1];

	free(sy);
	return retval;
}

void print_state (gsl_multifit_fdfsolver *s, int PRINTCOEFFS);

int main (int argc, char **argv)
{
	int KILL_CR = 0;
	int PRINTCOEFFS = 0;
	int PRINTDATA = 1;
	int USEMAG = 0;
	int AUTOITER = 0;
	int MEDIAN = 0;
	int C3 = 0;

	int iters = 15;
	double sig_lo = 0.2;
	double sig_hi = 3.0;
	double sig_cr = 2.0;
	double xscale = 1.0;
	
	double sig_native, medval;
	int l;
	
	const gsl_multifit_fdfsolver_type *T;
	gsl_multifit_fdfsolver *s;
	gsl_multifit_function_fdf f;
	gsl_vector_view x;
	
	int ptsno, ptsno_orig;
	int status, i, j, iter = 0, colno;
	char *datafile;
	FILE *input;
	double rx, ry, rw;
	char line[255];
	
	double *X = NULL, *Y = NULL, *W = NULL;
	double *Xorig, *Yorig, *Worig, newY;
	struct data d;
	double *x_init;
	
	if (argc < 2) {
		printf ("\nUsage: ./sigclip [-o order] [-i iters] [-s lo hi] [--nocr] [--pc] [--pd] input.data \n\n");
		printf ("  -o order:   Legendre polynomial order (default: 2)\n");
		printf ("  -i iters:   sigma clipping iterations (default: 15)\n");
		printf ("  -s lo hi:   sigma clipping boundaries\n");
		printf ("  --nocr sig: remove 1-pixel cosmic rays and defects\n");
		printf ("  --autoiter: stop iterations automatically when converged\n");
		printf ("  --pc:       print coefficients\n");
		printf ("  --pd:       print data\n");
		printf ("  --xscale f  x-column multiplier (i.e. s->days)\n");
		printf ("  --median    renormalize the y-scale to the median before clipping\n");
		printf ("  --c3        print 3-column output\n");
		printf ("  --mag       output magnitudes instead of fluxes\n");
		printf ("  input:      2- or 3-column input: hjd, flux [, sigma]\n\n");
		exit (-1);
	}
	
	for (i = 1; i < argc; i++) {
		if (strcmp (argv[i], "--nocr") == 0) {
			KILL_CR = 1;
			i++; sig_cr = atof (argv[i]);
		}
		else if (strcmp (argv[i], "-o") == 0) {
			i++; Pl_order = atoi (argv[i]);
		}
		else if (strcmp (argv[i], "-i") == 0) {
			i++; iters = atoi (argv[i]);
		}
		else if (strcmp (argv[i], "--xscale") == 0) {
			i++; xscale = atof (argv[i]);
		}
		else if (strcmp (argv[i], "-s") == 0) {
			i++; sig_lo = atof (argv[i]);
			i++; sig_hi = atof (argv[i]);
		}
		else if (strcmp (argv[i], "--pc") == 0)
			PRINTCOEFFS = 1;
		else if (strcmp (argv[i], "--autoiter") == 0) {
			iters = 999;
			AUTOITER = 1;
		}
		else if (strcmp (argv[i], "--median") == 0)
			MEDIAN = 1;
		else if (strcmp (argv[i], "--pd") == 0)
			PRINTDATA = 1;
		else if (strcmp (argv[i], "--c3") == 0)
			C3 = 1;
		else if (strcmp (argv[i], "--mag") == 0)
			USEMAG = 1;
		else datafile = argv[i];
	}
	
	fprintf (stderr, "# Issued: ");
	for (i = 0; i < argc; i++)
		fprintf (stderr, "%s ", argv[i]);
	fprintf (stderr, "\n");
	
	/******************** These are the data to be fitted *******************/

	i = 0;
	input = fopen (datafile, "r");
	if (!input) {
		printf ("File %s cannot be found, aborting.\n", datafile);
		exit (-1);
	}

	while (fgets (line, 255, input)) {
		colno = sscanf (line, "%lf %lf %lf\n", &rx, &ry, &rw);
		if (colno == 3) {
			X = realloc (X, (i+1)*sizeof(*X)); X[i] = rx;
			Y = realloc (Y, (i+1)*sizeof(*Y)); Y[i] = ry;
			W = realloc (W, (i+1)*sizeof(*W)); W[i] = rw;
			i++;
		}
		else if (colno == 2) {
			X = realloc (X, (i+1)*sizeof(*X)); X[i] = rx;
			Y = realloc (Y, (i+1)*sizeof(*Y)); Y[i] = ry;
			W = realloc (W, (i+1)*sizeof(*W)); W[i] = 1.0;
			i++;
		}
		else continue;
	}
	ptsno = ptsno_orig = i;

	if (MEDIAN) {
		medval = median(Y, ptsno);
		fprintf (stderr, "# Median: %g\n", medval);
		for (i = 0; i < ptsno; i++)
			Y[i] /= medval;
	}

	/* Make a copy of the original data since there will be some culling: */
	Xorig = malloc (ptsno*sizeof(*Xorig));
	Yorig = malloc (ptsno*sizeof(*Yorig));
	Worig = malloc (ptsno*sizeof(*Worig));
	for (i = 0; i < ptsno; i++) {
		Xorig[i] = X[i];
		Yorig[i] = Y[i];
		Worig[i] = W[i];
	}

	/* Map input times to a unit interval: */
	for (i = 0; i < ptsno; i++)
		X[i] = (X[i]-Xorig[0])/(Xorig[ptsno-1]-Xorig[0]);

	fprintf (stderr, "# Standard deviation: %lf\n", sig_native = stdev (Y, i));
	fprintf (stderr, "# Original number of data points: %d\n", i);		

	/**************************************************************************/

	d.order = Pl_order;
	f.f = &Pl_f;
	f.df = &Pl_df;
	f.fdf = &Pl_fdf;
	f.p = Pl_order;
	f.params = &d;

	for (l = 0; l < iters; l++) {
		d.n = ptsno;
		d.x = X;
		d.y = Y;
		d.w = W;
		
		f.n = ptsno;
		
		x_init = malloc (Pl_order * sizeof (*x_init));
		for (i = 0; i < Pl_order; i++)
			x_init[i] = 1.0;
		
		x = gsl_vector_view_array (x_init, Pl_order);
		
		/*********************************************************************/
		
		T = gsl_multifit_fdfsolver_lmsder;
		s = gsl_multifit_fdfsolver_alloc (T, ptsno, Pl_order);
		gsl_multifit_fdfsolver_set (s, &f, &x.vector);
		
		/*********************************************************************/
		
		do {
			iter++;
			status = gsl_multifit_fdfsolver_iterate (s);
			if (status) break;

			status = gsl_multifit_test_delta (s->dx, s->x, 1e-4, 1e-4);
		} while (status == GSL_CONTINUE && iter < 999);

		print_state (s, PRINTCOEFFS);

		ptsno = cull (&X, &Y, &W, ptsno, s->x, -sig_lo*sig_native, sig_hi*sig_native);
		if (AUTOITER == 1 && ptsno == d.n) break;
		fprintf (stderr, "# Reduced data points:  %d\n", ptsno);
	}

	if (PRINTDATA) {
		double *Plpolys = malloc (Pl_order*sizeof(*Plpolys));
		double *F;

		F = malloc (ptsno_orig*sizeof(*F));
		for (i = 0; i < ptsno_orig; i++) {
			gsl_sf_legendre_Pl_array (Pl_order-1, (Xorig[i]-Xorig[0])/(Xorig[ptsno_orig-1]-Xorig[0]), Plpolys);
			F[i] = 0.0;
			for (j = 0; j < Pl_order; j++)
				F[i] += Plpolys[j]*gsl_vector_get (s->x, j);
		}
		free (Plpolys);

		/* Kill cosmic rays? */
		if (KILL_CR) {
			int ncr;
			double *rect = malloc (ptsno_orig*sizeof(*rect));
			double sig_new;
			for (i = 0; i < ptsno_orig; i++)
				rect[i] = Yorig[i]/F[i];
			sig_new = stdev (rect, i);
			fprintf (stderr, "# Corrected sigma: %lf\n", sig_new);
			fprintf (stderr, "# Cosmics removed: %d\n", ncr = kill_cr (&Yorig, ptsno_orig, sig_cr*sig_new));
			free (rect);
		}

		for (i = 0; i < ptsno_orig; i++) {
			if (USEMAG == 1)
				newY = -5./2.*log10(Yorig[i]/F[i]);
			else
				newY = Yorig[i]/F[i];
			
			if (C3 == 1)
				printf ("%10.8lf %10.8lf %10.8lf\n", Xorig[i]/xscale, newY, Worig[i]/F[i]);
			else
				printf ("%10.8lf %10.8lf %10.8lf %10.8lf %10.8lf %10.8lf %10.8lf\n", Xorig[i], Yorig[i], Worig[i], F[i], Xorig[i]/xscale, newY, Worig[i]/F[i]);
		}

		free (F);
		}

	gsl_multifit_fdfsolver_free (s);
	free (X); free (Xorig);
	free (Y); free (Yorig);
	free (W); free (Worig);

	return 0;
}

void print_state (gsl_multifit_fdfsolver *s, int PRINTCOEFFS)
{
	int i;

	if (PRINTCOEFFS == 0) return;
	
	for (i = 0; i < Pl_order; i++) {
		printf ("%18.9E", gsl_vector_get (s->x, i));
	}
	printf ("\n");
}

