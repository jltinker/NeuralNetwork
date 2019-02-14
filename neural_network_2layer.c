#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
//
#include <time.h>
#include "cutil.h"
//#include "nrutil.h"

#define c_on_H0 3000
#define PI 3.14159
#define OMEGA_M 0.29
#define RA_MIN 100
#define RA_MAX 260
#define Q0 2.0
#define Q1 -1.0
#define QZ0 0.1
#define COLLISION_ANGLE (62./3600.*3.14159/180.)
#define RAD (180/PI)
#define ind(i,j) (i*njack+j)
#define mabs(A) ((A) < 0.0 ? -(A) : (A))


#define NumLayers 2
#define NumTrain 2000
#define    NumHidden 8 
#define    NumOutput 1
#define    NumInput 8

float Input[NumTrain+1][NumInput+1] ; //numinput is free parameters
float InputError[NumTrain+1][NumInput+1] ; //numinput is free parameters
float Output[NumTrain+1][NumOutput+1] ; // numoutput is 1? (wp?)
float SumH[NumTrain+1][NumHidden+1] ; //numpattern is N_train
float SumH2[NumTrain+1][NumHidden+1] ; //numpattern is N_train
float SumO[NumTrain+1][NumOutput+1] ;
float SumDOW[NumHidden+1];
float DeltaO[NumOutput+1];
float DeltaH[NumHidden+1];
float DeltaH2[NumHidden+1];
float Hidden[NumTrain+1][NumHidden+1] ;
float Hidden2[NumTrain+1][NumHidden+1] ;
float WeightIH[NumInput+1][NumHidden+1] ;
float WeightIH2[NumInput+1][NumHidden+1] ;
float WeightHO[NumHidden+1][NumOutput+1] ;
float WeightHO2[NumHidden+1][NumHidden+1] ;
float DeltaWeightIH[NumInput+1][NumHidden+1];
float DeltaWeightIH2[NumInput+1][NumHidden+1];
float DeltaWeightHO[NumHidden+1][NumOutput+1];
float Target[NumTrain+1][NumOutput+1]; // training value of wp
float TargetError[NumTrain+1][NumOutput+1]; // training value of wp


/* local functions
 */
int network_train();
float activation_function(float x);
int input_training_data(char *fname, char *fnamep);
int initialize_weights();

int main(int argc, char **argv)
{
  if(argc<2)
    {
      fprintf(stderr,"./ANN target_filename param_filename > out\n");
      exit(0);
    }

  input_training_data(argv[1], argv[2]);
  initialize_weights();
  network_train();

}

int initialize_weights()
{
  int i,j,k;
  float smallwt=0.05;

  srand48(555);

    for( j = 1 ; j <= NumHidden ; j++ ) {         /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
        }
    }
    for( j = 1 ; j <= NumHidden ; j++ ) {         /* initialize WeightIH2 and DeltaWeightIH2 */
        for( i = 0 ; i <= NumHidden ; i++ ) {
            DeltaWeightIH2[i][j] = 0.0 ;
            WeightIH2[i][j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
        }
    }
    for( k = 1 ; k <= NumOutput ; k ++ ) {         /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
        }
    }
    return 0;

}

int input_training_data(char *fname, char *fnamep)
{
  FILE *fp;
  int i,j;
  float mean=0;

  fp = openfile(fname);
  if(filesize(fp)!=NumTrain)
    {
      fprintf(stderr,"Error! File size mismatch in training sample.\n");
      exit(0);
    }

  for(i=1;i<=NumTrain;++i)
    fscanf(fp,"%f %f",&Target[i][1],&TargetError[i][1]);
  fclose(fp);

  // let's do the log of that number
  for(i=1;i<=NumTrain;++i)
    {
      TargetError[i][1] /= Target[i][j];
      TargetError[i][1] = 1;
      Target[i][1] = log(Target[i][1]);
    }

  // let's divide out the mean
  for(i=1;i<=NumTrain;++i)
    mean+=Target[i][1];
  mean = mean/NumTrain;

  // if doing log, subtract mean
  for(i=1;i<=NumTrain;++i)
    Target[i][1]-=mean;
  

  // if doing log, don't need to do this division
  /*
  for(i=1;i<=NumTrain;++i)
    Target[i][1]/=mean;
  for(i=1;i<=NumTrain;++i)
    TargetError[i][1]/=mean;
  */

  fp = openfile(fnamep);
  if(filesize(fp)!=NumTrain)
    {
      fprintf(stderr,"Error! File size mismatch in training sample parameters.\n");
      exit(0);
    }

  for(i=1;i<=NumTrain;++i)
    for(j=1;j<=NumInput;++j)
    fscanf(fp,"%f",&Input[i][j]);

  // let's error trap for the big masses: if larger than 1000, go to ln
  for(i=1;i<=NumTrain;++i)
    for(j=1;j<=NumInput;++j)
      if(Input[i][j]>1000)Input[i][j]=log(Input[i][j]);

  fclose(fp);



  return 0;
}

float activation_function(float x)
{
  float alpha = 0;
  
  // lets try straight-up linear function
  //return x;

  // this is ReLU
  if(x<-3)return alpha*x;
  return x;
}
float activation_derivative(float x)
{
  float alpha = 0;
  // lets try straight-up linear function
  //return 1;

  // this is ReLU
  if(x<-3)return alpha;
  return 1;
}

int network_train()
{
  float eta=0.001, alpha=0.000;
  int i,p,k,j,niter=0, ip, prand[NumTrain+1];

  float Error;

  Error = 10*NumTrain;
  while (Error>.0) {

    // get random list of training sample (use bootstrap method)
    for(i=1;i<=NumTrain;++i)
      prand[i] = (int)(drand48()*NumTrain) + 1;

  Error = 0.0 ;
  for( ip = 1 ; ip <= NumTrain ; ip++ ) {         /* repeat for all the training patterns */

    //p = prand[ip];
    p = ip;

    for( j = 1 ; j <= NumHidden ; j++ ) {         /* compute hidden unit activations */
        SumH[p][j] = WeightIH[0][j] ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
        }
	//printf("SumH: %e\n",SumH[p][j]);
        Hidden2[p][j] = Hidden[p][j] = activation_function(SumH[p][j]);
	//printf("1 %d %e %e\n",j,SumH[p][j],Hidden[p][j]);
    }

    // add a second hidden layer
    if(NumLayers>1)
      {
	for( j = 1 ; j <= NumHidden ; j++ ) {         /* compute hidden unit activations */
	  SumH2[p][j] = WeightIH2[0][j] ;
	  for( i = 1 ; i <= NumHidden ; i++ ) {
            SumH2[p][j] += Hidden[p][i] * WeightIH2[i][j] ;
	  }
	  //printf("SumH2: %e\n",SumH2[p][j]);
	  Hidden2[p][j] = activation_function(SumH2[p][j]);
	  //printf("2 %d %e %e\n",j,SumH2[p][j],Hidden2[p][j]);
	}
      }

    for( k = 1 ; k <= NumOutput ; k++ ) {         /* compute output unit activations and errors */
        SumO[p][k] = WeightHO[0][k] ;
        for( j = 1 ; j <= NumHidden ; j++ ) {
            SumO[p][k] += Hidden2[p][j] * WeightHO[j][k] ;
        }
	//printf("SumO: %e\n",SumH[p][k]);
        Output[p][k] = activation_function(SumO[p][k]) ;
        //Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;
        Error +=(Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k])/
	  TargetError[p][k]/TargetError[p][k] ;
	if(niter==-10000) {
	  if(p>=1)printf("Target: %d %e %e %e %e\n",p,Target[p][k],Output[p][k], TargetError[p][k], WeightHO[1][1]);
	}
	//if(p>=1)printf("Target: %d %e %e %e %e\n",p,Target[p][k],Output[p][k], TargetError[p][k], WeightHO[1][1]);
        //DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;
        DeltaO[k] = (Target[p][k] - Output[p][k]) * activation_derivative(SumO[p][k]) ;
	//if(p>=1)printf("DeltaO %e\n",DeltaO[k]);
    }
    if(NumLayers==1)
      {
	for( j = 1 ; j <= NumHidden ; j++ ) {         /* 'back-propagate' errors to hidden layer */
	  SumDOW[j] = 0.0 ;
	  for( k = 1 ; k <= NumOutput ; k++ ) {
            SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
	  }
	  //DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
	  DeltaH[j] = SumDOW[j] * activation_derivative(SumH[p][j]) ;
	  //if(p>=1)printf("DeltaH %d %e %e %e\n",j,DeltaH[j],activation_derivative(SumH[p][j]),SumH[p][j]);
	}
      }

    if(NumLayers>1)
      {
	for( j = 1 ; j <= NumHidden ; j++ ) {         /* 'back-propagate' errors to hidden layer */
	  SumDOW[j] = 0.0 ;
	  for( k = 1 ; k <= NumOutput ; k++ ) {
            SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
	  }
	  //DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
	  DeltaH2[j] = SumDOW[j] * activation_derivative(SumH2[p][j]) ;
	  //if(p>=1)printf("DeltaH %d %e %e %e\n",j,DeltaH2[j],activation_derivative(SumH2[p][j]),SumH2[p][j]);
	}
	for( j = 1 ; j <= NumHidden ; j++ ) {         /* 'back-propagate' errors to hidden layer */
	  SumDOW[j] = 0.0 ;
	  for( k = 1 ; k <= NumHidden ; k++ ) {
            SumDOW[j] += WeightIH2[j][k] * DeltaH2[k] ;
	  }
	  //DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
	  DeltaH[j] = SumDOW[j] * activation_derivative(SumH[p][j]) ;
	  //if(p>=1)printf("DeltaH %d %e %e %e\n",j,DeltaH[j],activation_derivative(SumH[p][j]),SumH[p][j]);
	}
	//exit(0);
      }

    for( j = 1 ; j <= NumHidden ; j++ ) {         /* update weights WeightIH */
        DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
        WeightIH[0][j] += DeltaWeightIH[0][j] ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
            WeightIH[i][j] += DeltaWeightIH[i][j] ;
	    //printf("1 %d %d %e %e\n",i,j,WeightIH[i][j],DeltaWeightIH[i][j]);
        }
    }
    for( j = 1 ; j <= NumHidden ; j++ ) {         /* update weights WeightIH2 */
        DeltaWeightIH2[0][j] = eta * DeltaH2[j] + alpha * DeltaWeightIH2[0][j] ;
        WeightIH2[0][j] += DeltaWeightIH2[0][j] ;
        for( i = 1 ; i <= NumHidden ; i++ ) {
            DeltaWeightIH2[i][j] = eta * Hidden[i][j] * DeltaH2[j] + alpha * DeltaWeightIH2[i][j];
            WeightIH2[i][j] += DeltaWeightIH2[i][j] ;
	    printf("2 %d %d %e %e\n",i,j,WeightIH2[i][j],DeltaWeightIH2[i][j]);
        }
    }
    exit(0);
    for( k = 1 ; k <= NumOutput ; k ++ ) {         /* update weights WeightHO */
        DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
        WeightHO[0][k] += DeltaWeightHO[0][k] ;
        for( j = 1 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = eta * Hidden2[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
            WeightHO[j][k] += DeltaWeightHO[j][k] ;
	    //if(j==1)printf("DeltaH0 %d %e %e\n",p,WeightHO[j][k],DeltaWeightHO[j][k]);
        }
    }

    //exit(0);
  }
  Error = sqrt(Error/NumTrain);
  //if(niter==1000)exit(0);
  ++niter;
  if(niter%1000 == 1 || niter<1000) { 
    printf("Iter: %d Error: %e %f\n",niter,Error,Error);
    fflush(stdout);
  }
  if(Error<0.06)
    {
      k = 1;
      for(p=1;p<=NumTrain;++p)
	printf("Target: %d %e %e %e\n",p,exp(Target[p][k]),exp(Output[p][k]), 
	       TargetError[p][k]);
      exit(0);
    }

  //if(niter==1000)exit(0);
  }


  return 0;

}
