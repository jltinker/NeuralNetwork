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


#define NumLayers 4
#define NumTrain 2000
#define    NumHidden 16 
#define    NumOutput 1
#define    NumInput 8

float Input[NumTrain+1][NumInput+1] ; //numinput is free parameters
float InputError[NumTrain+1][NumInput+1] ; //numinput is free parameters
float Output[NumTrain+1][NumOutput+1] ; // numoutput is 1? (wp?)
float SumH[NumLayers+1][NumTrain+1][NumHidden+1] ; //numpattern is N_train
float SumO[NumTrain+1][NumOutput+1] ;
float SumDOW[NumHidden+1];
float DeltaO[NumOutput+1];
float DeltaH[NumLayers+1][NumHidden+1];
float Hidden[NumLayers+1][NumTrain+1][NumHidden+1] ;
float WeightIH[NumLayers+1][NumHidden+1][NumHidden+1] ; //assuming NumHidden>=NumInput
float WeightHO[NumHidden+1][NumOutput+1] ;
float DeltaWeightIH[NumLayers+1][NumHidden+1][NumHidden+1]; // assuming NumHidden>=NumInput
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
  float smallwt=0.4;

  srand48(555);

  /* Let's assume that NumHidden and NumInput is the same
     throughout usage.
  */
  for(k=1;k<=NumLayers;++k) {
    for( j = 1 ; j <= NumHidden ; j++ ) {         /* initialize WeightIH and DeltaWeightIH */
      for( i = 0 ; i <= NumHidden ; i++ ) {
	DeltaWeightIH[k][i][j] = 0.0 ;
	WeightIH[k][i][j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
      }
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
      TargetError[i][1] /= Target[i][1];
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
  float eta=0.002, alpha=0.001;
  int n,i,p,k,j,niter=0, ip, prand[NumTrain+1];

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

    for(n = 1; n<= NumLayers; ++n) {
      for( j = 1 ; j <= NumHidden ; j++ ) {         /* compute hidden unit activations */
        SumH[n][p][j] = WeightIH[n][0][j] ;
        for( i = 1 ; i <= NumHidden ; i++ ) {
	  if(n==1 && i<=NumInput) 
            SumH[n][p][j] += Input[p][i] * WeightIH[n][i][j] ; // from input layerer
	  else 
	    SumH[n][p][j] += Hidden[n-1][p][i]*WeightIH[n][i][j]; // from hidden layer
        }
        Hidden[n][p][j] = activation_function(SumH[n][p][j]);
	//printf("%d %d %e %e\n",n,j,SumH[n][p][j],Hidden[n][p][j]);
      }
    }

    for( k = 1 ; k <= NumOutput ; k++ ) {         /* compute output unit activations and errors */
      SumO[p][k] = WeightHO[0][k] ;
      for( j = 1 ; j <= NumHidden ; j++ ) {
	SumO[p][k] += Hidden[NumLayers][p][j] * WeightHO[j][k] ;
      }
      //printf("SumO: %e\n",SumH[p][k]);
      Output[p][k] = activation_function(SumO[p][k]) ;
      Error +=(Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k])/
	TargetError[p][k]/TargetError[p][k] ;
      if(niter==-10000) {
	if(p>=1)printf("Target: %d %e %e %e %e\n",p,Target[p][k],Output[p][k], TargetError[p][k], WeightHO[1][1]);
      }
      DeltaO[k] = (Target[p][k] - Output[p][k]) * activation_derivative(SumO[p][k]) ;
      //if(p>=1)printf("DeltaO %e\n",DeltaO[k]);
    }
    
    /* 'back-propagate' errors to hidden layer from output layer 
     */    
    for( j = 1 ; j <= NumHidden ; j++ ) {         
      SumDOW[j] = 0.0 ;
      for( k = 1 ; k <= NumOutput ; k++ ) {
	SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
      }
      DeltaH[NumLayers][j] = SumDOW[j] * activation_derivative(SumH[NumLayers][p][j]) ;
      //if(p>=1)printf("DeltaH %d %e %e %e\n",j,DeltaH[NumLayers][j],
      //     activation_derivative(SumH[NumLayers][p][j]),SumH[NumLayers][p][j]);
    }


    /* back-propagate between the hidden layers
     */
    for(n=NumLayers-1; n>=1 ; --n ) {
      for( j = 1 ; j <= NumHidden ; j++ ) {         /* 'back-propagate' errors to hidden layer */
	SumDOW[j] = 0.0 ;
	for( k = 1 ; k <= NumHidden ; k++ ) {
	  SumDOW[j] += WeightIH[n+1][j][k] * DeltaH[n+1][k] ;
	}
	DeltaH[n][j] = SumDOW[j] * activation_derivative(SumH[n][p][j]) ;
	//if(p>=1)printf("DeltaH %d %e %e %e\n",j,DeltaH[n][j],
	//	       activation_derivative(SumH[n][p][j]),SumH[n][p][j]);
      }
    }
    //exit(0);

    
    /* Update the weights in the hidden layers
     */
    for(n=1;n<=NumLayers;++n){
      for( j = 1 ; j <= NumHidden ; j++ ) {         /* update weights WeightIH */
        DeltaWeightIH[n][0][j] = eta * DeltaH[n][j] + alpha * DeltaWeightIH[n][0][j] ;
        WeightIH[n][0][j] += DeltaWeightIH[n][0][j] ;
        for( i = 1 ; i <= NumHidden ; i++ ) {
	  if(n==1 && i<=NumInput) 
	    DeltaWeightIH[n][i][j] = eta * Input[p][i] * DeltaH[n][j] + alpha * DeltaWeightIH[n][i][j];
	  else
	    DeltaWeightIH[n][i][j] = eta * Hidden[n-1][p][i] * DeltaH[n][j] + alpha * DeltaWeightIH[n][i][j];
	  WeightIH[n][i][j] += DeltaWeightIH[n][i][j] ;
	  //if(n==2)
	  //printf("%d %d %d %e %e\n",n,i,j,WeightIH[n][i][j],DeltaWeightIH[n][i][j] );
        }
      }
    }
    //      exit(0);

    /* Update the weights in the output layer
     */
    for( k = 1 ; k <= NumOutput ; k ++ ) {         /* update weights WeightHO */
      DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
      WeightHO[0][k] += DeltaWeightHO[0][k] ;
      for( j = 1 ; j <= NumHidden ; j++ ) {
	DeltaWeightHO[j][k] = eta * Hidden[NumLayers][p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
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
  if(Error<0.01)
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
