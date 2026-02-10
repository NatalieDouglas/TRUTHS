#ifndef NADIM_H
#define NADIM_H

#define NWBANDS 421
#define MAX_LINE_LEN 5000
#define NANGLES 500
#define NWAVELS 500

void leaftwo_( double *, double *, double *, double *, double *, double *, double * );
void nadimbrf_( float *, float *, int *, float *, float *, int *, float *, float *, float *, float *, float *, float *, float * );
void nadimbrfe_( float *, float *, int *, float *, float *, int *, float *, float *, float *, float *, float *, 
												float *, float * , float *, float *,  float * );
void energie_( float *, float *,  float *, float *,  float *  );
int prospect_monochromatic( float, double, double, double, double, double, double *, double * );
int price_soil_monochromatic( float, float, float, float, float, float *, float *, float *, float *, float * );

#endif /*NADIM_H*/
