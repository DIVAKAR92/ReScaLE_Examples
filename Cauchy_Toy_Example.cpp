#include <Rcpp.h> 
#include <stdlib.h>
#include <vector>
using namespace Rcpp; 
using namespace std;

// ============= dataset being used in this example ============================== //
static  std::vector<double> data{2.65226687,1.27648783,1.61011759,1.27433040,0.08721209};
static int N = data.size();
//==================================================================================//

// ========= gradient of the log-likelihood function (i.e. drift function ========= //
// Recall that it uses a standard cauchy density as a prior 
double drift (double x){
  double result = 0;
  // sum the gradient corresponding to each data point in a loop
  for (int i =0; i<N; i++){
    result = result + 2*(data[i] - x)/(1 + pow((data[i] - x),2));
  }
  // sum the gradient corresponding to the prior 
  result = result - 2*x/(1+x*x);
  return result; 
}
//==================================================================================//

// =============== second derivative function of the log-likelihood function ===== //
double divergence (double x){
  double result = 0;
  // sum the second derivative function for each data point in a loop
  for (int i =0; i<N; i++){
    result = result - 2*(1-pow((data[i] - x),2))/pow((1 + pow((data[i] - x),2)),2);
  }
  // sum the second derivative corresponding to standard cauchy prior
  result = result - 2*(1-x*x)/pow((1+x*x),2);
  return result; 
}
//==================================================================================//

// ================== kappa function for this toy example ========================= //
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double kappa(double x){
  // sum of the squared drift and the divergence which is then halved 
  double result = 0.5*(pow(drift(x),2) + divergence(x));
  // -2.379829 is the global lower bound of the term 
  result =  result + 2.379829;
  return result;
} 
//==================================================================================//

// ================== upper bound function ========================================//
double M_Cauchy(){
  return 14.0;
}
//==================================================================================//

// ================== un-normalised posterior density ==============================//
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
std::vector<double> unnormalised_posterior(std::vector<double> x){
  double prod;
  int Nx = x.size();
  std::vector<double> result(Nx);
  for (int j=0; j < Nx; j++){
    prod = 1;
    for(int i=0; i <N; i++){
      prod = prod*1/(1+pow((data[i]-x[j]),2));
    }
    prod = prod/(1+x[j]*x[j]);
    result[j] = prod; 
  }
  return result;
}  