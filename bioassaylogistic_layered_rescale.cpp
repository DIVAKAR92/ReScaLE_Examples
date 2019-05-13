#include <Rcpp.h> 
#include <Rmath.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#define a 1.243707
#define alpha 1.088870
#define lambda 1.233701
using namespace std;
using namespace Rcpp; 

static double DIM =2;
static std::vector<std::vector<double>> data;
static double N_data; //data.size();

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
void set_ndata(double x){
  N_data = x;
}

//============= construction of kappa function and its bounding function ======== //

// Function which calculates and returns full first and second derivatives of the 
// logistic regression model on the bioassay data. Note that the data has been hard
// coded in this example to enhance efficiency of the code. 
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
vector<double> bioassay_derivaties(vector<double> x){
  /*
  Input:
  --------------------------------------------------------------
  x -- the point at which the calculation of derivatives is done
  --------------------------------------------------------------
  */
  // X is the normalised covariate
  vector<double> X = {-0.56047741,-0.13633234  ,0.05301813,  0.64379162};
  // N is the total number of people considered in the each group
  vector<double> N ={5,5,5,5};
  // n is the number of people who respond positive out of N 
  vector<double> n = {0, 1, 3, 5};
  // A container to store the derivatives 
  vector<double> res(5, 0); double plogi;
  // first and the second derivatives initially set to zero 
  double f_x=0, f_y=0, f_xx=0, f_xy=0, f_yy=0;
  // looping over entire data and summing the derivatives
  for(int i=0; i<4; i++){
    // calculating the probability corresponding to a given X
    plogi = exp(x[0]+x[1]*X[i])/(1+exp(x[0]+x[1]*X[i]));
    // calculation of the first derivatives 
    f_x += n[i] - N[i]*plogi + 0.25*(-2*x[0]/100)/pow(1+x[0]*x[0]/100, 1);
    f_y += (n[i] - N[i]*plogi)*X[i] + 0.25*(-2*x[1]/6.25)/pow(1+x[1]*x[1]/6.25, 1);
    // Calculation of the hessian matrix 
    f_xx += -N[i]*plogi*(1-plogi) + 0.25*(2/100)*(-1+x[0]*x[0]/100)/pow(1+x[0]*x[0]/100, 2);
    f_xy += -N[i]*plogi*(1-plogi)*X[i];
    f_yy += -N[i]*plogi*(1-plogi)*X[i]*X[i] + 0.25*(2/6.25)*(-1+x[1]*x[1]/6.25)/pow(1+x[1]*x[1]/6.25, 2);
  }
  // store all derivatives in a vector and return for later uses 
  res[0] = f_x; res[1] = f_y; res[2] = f_xx; res[3] = f_xy; res[4] = f_yy;
  return res;
}

// Function to calculate the hazard rate for a logistic regression model on the 
// bioassay data
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double bioassay_kappa_trans(vector<double> z){
  // An estimate of the posterior mode
  vector<double> map = {-0.15348 ,  9.23450  };
  // A transformation matrix which is used to transform the posterior distribution
  vector<vector<double>> T = {{0.7118026,0},{0.5456446, 5.253438}};
  vector<double> xt(2,0);
  // A scalar transformation factor, I set it equal to one since I have been using the 
  // square root of the variance covarinace matrix to transform the space 
  double tf = sqrt(100);
  // transformation of the point z 
  xt[0] = (T[0][0]*z[0])/tf + map[0];
  xt[1] = (T[1][0]*z[0] + T[1][1]*z[1])/tf + map[1];
  // calculate all first and second derivatives 
  vector<double> d = bioassay_derivaties(xt);
  // res[0] = f_x; res[1] = f_y; res[2] = f_xx; res[3] = f_xy; res[4] = f_yy;
  // calculations of all derivatives under the transformation
  double f_z1 = d[0]*T[0][0] + d[1]*T[1][0];
  double f_z2 = d[0]*T[0][1] + d[1]*T[1][1];
  double f_zz1 = (d[2]*T[0][0] + d[3]*T[1][0])*T[0][0] +
    (d[3]*T[0][0] + d[4]*T[1][0])*T[1][0];
  double f_zz2 = (d[2]*T[0][1] + d[3]*T[1][1])*T[0][1] +
    (d[3]*T[0][1] + d[4]*T[1][1])*T[1][1];
  // calculation of the kappa function 
  double res = pow((f_z1),2) + pow((f_z2),2) + f_zz1 + f_zz2;
  // I have added a constant based on numerical calculation such that 
  // the kappa function remains non-negative
  return ((res/(2)) + 3.7)/(pow(tf, 2));
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double M_bioassay (std::vector<double> lower, std::vector<double> upper){
  double dim = lower.size();
  std::vector<double> v1(2),v2(2),v3(2),v4(2);
  // get the vertices of the hypercube 
  v1[0] = lower[0]; v1[1] = lower[1];
  v2[0] = upper[0]; v2[1] = lower[1];
  v3[0] = upper[0]; v3[1] = upper[1];
  v4[0] = lower[0]; v4[1] = upper[1];
  std::vector<double> values(pow(2,dim),0);
  // calculate the value of kappa at each vertex 
  values[0] = bioassay_kappa_trans(v1); values[1] = bioassay_kappa_trans(v2);
  values[2] = bioassay_kappa_trans(v3); values[3] = bioassay_kappa_trans(v4);
  // find the maximum value 
  double max_val = 0;
  std::vector<double>::iterator it = max_element(values.begin(),values.end());
  max_val = *(it);
  return max_val+0.12;//0.56;//0.001;//1.5; //25.5
}

//============= kappa (I call it PHI_mult) and its bound function M ==========//

//============= DO NOT CHANGE ITS NAME ==================================//
// Wrapper function for the kappa -- the non-negative hazard rate 
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double PHI_mult (std::vector<double> x, 
                 std::vector<double> lower, 
                 std::vector<double> upper){
  return bioassay_kappa_trans(x);
}

//============= DO NOT CHANGE ITS NAME ==================================//
// Wrapper function for the bounds of the kappa function
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double M_mult (std::vector<double> lower, std::vector<double> upper){
  return M_bioassay(lower, upper); 
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
int search_elements_2 (std::vector<double> vec, double val){
  vector <double>::iterator i = find (vec.begin (),vec.end (), val);
  return distance (vec.begin (), i);
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
int search_elements (std::vector<double> vec, double val ){
  int initial = 0, final =  vec.size() - 1;
  int center, location=0;
  while (initial <= final ){
    center = (initial + final)/2;
    if ((vec[center] < val)&(vec[center+1] > val)){
      location = center;
      break;
    }
    if ((vec[center] > val)||(vec[center]==val)){
      final = center -1;
    }
    if ((vec[center] < val)||(vec[center]==val)){
      initial = center + 1;
    }
  }
  return location;
}


// [[Rcpp::export]]
void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}

// returns the maximum of two numbers 
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double maximum(double x, double y){
  double max = (x > y) ? x : y;
  return max;
}


// g_n function is defined as it has been defined in the paper by Burq and Jones 
// for the simulation of first time exit of a Brownian motion
double g_n (double n, double t){
  double result;
  result = n/std::sqrt(2*PI*pow(t,3))*std::exp(-pow(n,2)/(2*t));
  return result;
}

// returns the sign of a double variable
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double sign (double x){
  double sign = (x < 0) ? -1 : 1;
  return sign;
}

// f_n is the sum of alternate signs of f_n defined as per Burq and Jones paper
double f_n (double n, double t){
  double result = 0;
  for(int i = -n; i <= n; i++){
    result = result + pow(-1,i)*g_n(1+2*i,t);
  }
  return result; 
}

// Exit time distribution: implemented in the similar way as in Burq and Jones
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double exit_time_bm(double L){
  bool accepted = false;
  double V;
  while (accepted == false){
    double U = runif(1)[0];
    V = R::rgamma(alpha , 1/lambda);
    double Y = a*(R::dgamma(V, alpha,1/lambda,0))*U;
    double n = maximum(ceil(V*0.275),3);
    while (sign((Y-f_n(n,V))*(Y - f_n(n+1,V)))==-1){
      n = n+1;
    }
    if ( Y <= f_n(n+1,V)){
      accepted = true;
    }
  }
  return pow(L,2)*V; 
}


double psi (double j, double s, double x, double t, double y, double L, double U){
  double result;
  result = ((2*abs(U-L)*j - (x-L))/(x-L))*exp(-2*abs(U-L)*j*(abs(U-L)*j - (x-L))/(t-s));
  return result;  
}

double chi (double j, double s, double x, double t, double y, double L, double U){
  double result;
  result = ((2*abs(U-L)*j + (x-L))/(x-L))*exp(-2*abs(U-L)*j*(abs(U-L)*j + (x-L))/(t-s));
  return result;  
}

double sigma (double j, double s, double x, double t, double y, double L, double U){
  double result;
  result = exp(-(2/(t-s))*(j*(U-L)+(L-x))*(j*(U-L)+(L-y))) +
    exp(-(2/(t-s))*(j*(U-L)-(U-x))*(j*(U-L)-(U-y)));
  return result;
}

double phi (double j, double s, double x, double t, double y, double L, double U){
  double result;
  result = exp(-(2*j/(t-s))*(j*pow((U-L),2)+(U-L)*(x-y))) +
    exp(-(2*j/(t-s))*(j*pow((U-L),2)-(U-L)*(x-y)));
  return result;
}

double denominator(double s, double W_s, double t, double W_t, double W_tau){
  double result;
  result = 1 - exp(-2*(W_s - W_tau)*(W_t - W_tau)/(t-s));
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double p_lower (double n, double s, double W_s, double q, double W_q, double tau,
                double W_tau, double L, double U){
  double n0 = ceil(sqrt((tau-q)+pow((U-W_tau),2))/(2*abs(U-W_tau)));
  double sum1 = 1, sum2 = 1;
  double deno = denominator(s,W_s,q,W_q,W_tau);
  for(int j = 1; j<=(n+1); j++){
    sum1 = sum1 - sigma(j,s,W_s,q,W_q,L,U);
  }
  for(int j = 1; j<=(n); j++){
    sum1 = sum1 + phi(j,s,W_s,q,W_q,L,U);
  }
  for(int j = 1; j<=(n+n0+1); j++){
    sum2 = sum2 - psi(j,q,W_q,tau,W_tau,L,U);
  }
  for(int j = 1; j<=(n+n0); j++){
    sum2 = sum2 + chi(j,q,W_q,tau,W_tau,L,U);
  }
  // Rcpp::Rcout << " sum1 = " << sum1 << " sum2 = " << sum2/deno << endl;
  double result=sum1*sum2/deno;
  if (result < 0){ result = 0;}
  else if (result > 1) { result = 1;}
  return (result);
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double p_upper (double n, double s, double W_s, double q, double W_q, double tau,
                double W_tau, double L, double U){
  double n0 = ceil(sqrt((tau-q)+pow((U-W_tau),2))/(2*abs(U-W_tau)));
  double sum1 = 1, sum2 = 1;
  double deno = denominator(s,W_s,q,W_q,W_tau);
  for(int j = 1; j<=(n); j++){
    sum1 = sum1 - sigma(j,s,W_s,q,W_q,L,U);
  }
  for(int j = 1; j<=(n); j++){
    sum1 = sum1 + phi(j,s,W_s,q,W_q,L,U);
  }
  for(int j = 1; j<=(n+n0); j++){
    sum2 = sum2 - psi(j,q,W_q,tau,W_tau,L,U);
  }
  for(int j = 1; j<=(n+n0); j++){
    sum2 = sum2 + chi(j,q,W_q,tau,W_tau,L,U);
  }
  // Rcpp::Rcout << " sum1 = " << sum1 << " sum2 = " << sum2/deno << endl;
  double result=sum1*sum2/deno;
  if (result < 0){ result = 0;}
  else if (result > 1) { result = 1;}
  return (result);
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double bessel_proposal(double s, double W_s, double tau, double W_tau, double L,
                       double U, double q){
  double sd, b1, b2, b3, W_q;
  double C = W_tau > W_s ? 1 : 0;
  sd = sqrt(abs(tau-q)*abs(q-s)/pow((tau-s),2));
  b1 = rnorm(1,0,sd)[0];
  b2 = rnorm(1,0,sd)[0];
  b3 = rnorm(1,0,sd)[0];
  W_q = W_tau + pow(-1,C)*sqrt((tau-s)*(pow((abs(W_s-W_tau)*(tau-q)/pow((tau-s),1.5)+ b1),2)
                                          +(pow(b2,2))+(pow(b3,2))));
  return W_q;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
std::vector<double> exit_time_barrier(double s, double W_s, double L, double U){
  double discretization = 0.001;
  double W_t = W_s, t = s;
  while((W_t < U) & (W_t > L)){
    t = t + discretization;
    W_t = W_t + rnorm(1, 0, sqrt(discretization))[0];
  }
  double crossing_barrier;
  if(W_t > U){crossing_barrier = U;}
  else{crossing_barrier = L;}
  std::vector<double> result(2);
  result[0] = t; result[1] = crossing_barrier;
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double bool_bessel(double s, double W_s, double q, double W_q, double tau,
                   double W_tau, double L, double U){
  double u = runif(1)[0];
  double result, k = 1;
  
  if (W_tau == U){
    double lower = -U, upper = -L;
    W_s = -W_s; W_q = -W_q; W_tau = -W_tau; L = lower; U = upper;
  }
  while((p_lower(k,s,W_s,q,W_q,tau,W_tau,L,U) < u) & (u < p_upper(k,s,W_s,q,W_q,tau,W_tau,L,U))){
    // Rcpp::Rcout <<" p_low "<< p_lower(k,s,W_s,q,W_q,tau,W_tau,L,U) << "p_up " 
    //            << p_upper(k,s,W_s,q,W_q,tau,W_tau,L,U) << endl;
    k = k + 1;
  }
  if (u <= p_lower(k,s,W_s,q,W_q,tau,W_tau,L,U)){
    result = 1;
  }else{
    result = 0;
  }
  return result; 
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
double bessel_bridge_pos(double s, double W_s, double tau, 
                         double W_tau, double L, double U, double q){
  double W_q, check; bool accept = false;
  while (accept != true){
    do{
      W_q = bessel_proposal(s,W_s,tau,W_tau,L,U,q);
      // Rcpp::Rcout << " W_q " << W_q << endl;
    } while ((W_q > U) | (W_q < L));
    check = bool_bessel(s,W_s,q,W_q,tau,W_tau,L,U);
    // Rcpp::Rcout << " check " << check << endl;
    if(check == 1){
      accept = true;
    }
  }
  return W_q;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
std::vector<double> check_fun(double s, double W_s, double L,
                              double U, double n, double q){
  std::vector<double> res(n);
  double tau,W_tau, u,z, theta = (U-L)/2;
  for(int i = 0; i < n; i++){
    tau = s + exit_time_bm(theta);
    u = runif(1)[0];
    z = (u < 0.5) ? 1 : 0;
    W_tau = z*L + (1-z)*U;
    if(tau < q){
      res[i] = W_tau + rnorm(1,0,sqrt(q-tau))[0];
    }else{
      res[i] = bessel_bridge_pos(s,W_s,tau,W_tau,L,U,q);
    }
  }
  return res;
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double S_d1_upper(double k, double s, double x, double t, double y, double L, double U){
  double result = 1, sum = 0;
  // double deno = 1 - exp(-2*(x-min(L,U))*(y-min(L,U))/(t-s));
  for(double j=1; j<=k; j++){
    sum = sum - sigma(j,s,x,t,y,L,U) + phi(j,s,x,t,y,L,U);
  }
  result = result + sum;
  return result;
}
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double S_d1_lower(double k, double s, double x, double t, double y, double L, double U){
  double result;
  // double deno = 1 - exp(-2*(x-min(L,U))*(y-min(L,U))/(t-s));
  result = S_d1_upper(k,s,x,t,y,L,U) - sigma(k+1,s,x,t,y,L,U);
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double trunc_prob (double x){
  double result;
  if (x > 1){
    result = 1;
  }else if (x < 0){
    result = 0;
  }else{
    result = x;
  }
  return result;
}

double bool_const_bb(double s, double W_s, double q, double W_q, double t, 
                     double W_t, double L, double U){
  double check;
  double k=1,u=runif(1)[0];
  double p_l_1 = trunc_prob(S_d1_lower(k,s,W_s,q,W_q,L,U)), p_l_2 = trunc_prob(S_d1_lower(k,q,W_q,t,W_t,L,U));
  double p_u_1 = trunc_prob(S_d1_upper(k,s,W_s,q,W_q,L,U)), p_u_2 = trunc_prob(S_d1_upper(k,q,W_q,t,W_t,L,U));
  while((p_l_1*p_l_2<u)& (u<p_u_1*p_u_2)){
    k = k+1;
    p_l_1 = trunc_prob(S_d1_lower(k,s,W_s,q,W_q,L,U)), p_l_2 = trunc_prob(S_d1_lower(k,q,W_q,t,W_t,L,U));
    p_u_1 = trunc_prob(S_d1_upper(k,s,W_s,q,W_q,L,U)), p_u_2 = trunc_prob(S_d1_upper(k,q,W_q,t,W_t,L,U));
  }if(u <p_l_1*p_l_2){
    check = 1;
  }else{
    check = 0;
  }
  return check;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double brownian_bridge (double s, double W_s, double t, double W_t, double q){
  double W_q;
  double mean = W_s + ((q-s)/(t-s))*(W_t-W_s);
  double sd = sqrt((q-s)*(t-q)/(t-s));
  W_q = rnorm(1,mean,sd)[0];
  return W_q;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double constr_brow_bridge (double s, double W_s, double t, double W_t,
                           double L, double U, double q){
  bool accept = false;
  double W_q; double check;
  while(accept != true){
    do{
      W_q = brownian_bridge(s,W_s,t,W_t,q);
    } while ((L > W_q) | (W_q > U));
    check = bool_const_bb(s,W_s,q,W_q,t,W_t,L,U);
    if(check==1){
      accept = true;
    }
  }
  return W_q;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double bessel_inter_point(double s, double W_s, double t, double W_t, double tau, 
                          double W_tau, double L, double U, double q){
  double W_q;
  if(t==tau){
    W_q = bessel_bridge_pos(s,W_s,tau,W_tau,L,U,q);
  }else{
    W_q = constr_brow_bridge(s,W_s,t,W_t,L,U,q);
  }
  return W_q;
}


// Function returns the index of the last element which is less than given node/element
// in an increasing sequence of vector. 

int index_last_less_ele (std::vector<double> seq, double node){
  int i = 0;
  while (seq[i] < node){
    i++;
  }
  int two_closet = i-1;
  return two_closet;
}



double regenerate_pos(double s, double W_s, double t, double W_t,
                      double tau, double W_tau, double lower, double upper,
                      double reg_time, double t_kill){
  double unif = runif(1)[0]; double result;
  double prob = 1 - 1/(1+t_kill);
  if(unif < prob){
    result = bessel_inter_point(s,W_s,t,W_t,tau,W_tau,lower,upper,reg_time);
  }else{
    result = rnorm(1,0,6)[0];
  }
  return result;
}



// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
std::vector<double> intermediate_pos_sim(double s, double W_s, double t, double W_t, double tau, 
                                         double W_tau, double L, double U, std::vector<double> q){
  int N = q.size();
  std::vector<double> pos(N), POS(N+2), T(N+2);
  T[0] = s; T[N+1] = t; POS[0] = W_s; POS[N+1] = W_t;
  for(int i = 0; i < N; i++){
    T[i+1] =  q[i];
    POS[i+1] = bessel_inter_point(T[i],POS[i],T[N+1],POS[N+1],tau,W_tau,L,U,T[i+1]);
    pos[i] = POS[i+1];
  }
  return pos;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> capture (double left, double right, std::vector<double> mesh){
  int left_index = search_elements(mesh,left) + 1;
  if (left_index == 1){
    left_index = 0;
  }
  int right_index = search_elements(mesh,right);
  int N = right_index - left_index + 1;
  std::vector<double> result;
  for(int i = 0; i <= N; i++){
    if((left < mesh[i+left_index]) & (mesh[i+left_index] < right)){
      result.push_back(mesh[i+left_index]);
    }
  }
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> capture_2(double left, double right, std::vector<double> mesh){
  int subsetStartIdx = search_elements(mesh,left) + 1;
  int subsetEndIdx = search_elements(mesh,right) + 1;
  vector<double>::iterator subsetStartIter = mesh.begin() + subsetStartIdx;
  vector<double>::iterator subsetEndIter = mesh.begin() + subsetEndIdx;
  return std::vector<double> (subsetStartIter, subsetEndIter);
}
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> skeleton_at_given_mesh(std::vector<double> time,
                                           std::vector<double> pos,
                                           std::vector<double> l_layer,
                                           std::vector<double> u_layer,
                                           std::vector<double> tau,
                                           std::vector<double> W_tau,
                                           std::vector<double> mesh){
  int size = time.size();
  std::vector<double> RT, RP, result;
  for(int i=0; i < (size-1); i++){
    if(time[i] == time[i+1]){
      continue;
    }else{
      RT = capture(time[i],time[i+1],mesh);
      RP = intermediate_pos_sim(time[i],pos[i],time[i+1],pos[i+1],tau[i],W_tau[i],l_layer[i],u_layer[i],RT);
      result.insert(result.end(),RP.begin(),RP.end());
      RP.clear(); RT.clear();
    }
  }
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> skeleton_at_given_mesh_new(std::vector<double> time,
                                               std::vector<double> pos,
                                               std::vector<double> l_layer,
                                               std::vector<double> u_layer,
                                               std::vector<double> tau,
                                               std::vector<double> W_tau,
                                               std::vector<double> mesh){
  int size = mesh.size();
  std::vector<double> result(size);
  result[0] = pos[0];
  int li =0; int ri =0;
  for(int i=1; i < (size); i++){
    li = search_elements(time, mesh[i]); ri = li +1;
    result[i] = bessel_inter_point(time[li],pos[li],time[ri],pos[ri],tau[li],W_tau[li],
                                   l_layer[li],u_layer[li],mesh[i]);
    // Rcout << i << " ";
  }
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> skeleton_at_given_mesh_new_2(std::vector<double> time,
                                                 std::vector<double> pos,
                                                 std::vector<double> l_layer,
                                                 std::vector<double> u_layer,
                                                 std::vector<double> tau,
                                                 std::vector<double> W_tau,
                                                 std::vector<double> mesh,
                                                 double f1 = 10000, double f2 =5.36){
  int size = mesh.size(); double ts = time.size();
  int lid =0; int rid = 0;
  std::vector<double> result(size);
  result[0] = pos[0]; // int k =0;
  int li =0; int ri =0; int tmp = 0;
  vector<double>::iterator start = time.begin();
  for(int i=1; i < (size); i++){
    lid = std::max(0.0,floor(i*f2-f1)); rid = std::min(floor(i*f2+f1),ts);
    tmp = search_elements(std::vector<double>(start+lid, start+rid), mesh[i]);
    li = lid + tmp; ri = li +1;
    if((time[li] < mesh[i]) & (mesh[i] < time[ri])){
      result[i] = bessel_inter_point(time[li],pos[li],time[ri],pos[ri],tau[li],W_tau[li],l_layer[li],u_layer[li],mesh[i]);
    }else{
      result[i] = pos[ri];
      // k = k + 1;
    }
  }
  // Rcout << k << " ";
  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> kill_function(std::vector<double> x){
  int N = x.size();
  std::vector<double> result(N);
  for(int i =0; i < N; i++){
    result[i] = (1/2.506628)*exp(-x[i]*x[i]/2)*(x[i]*x[i]);
  }
  return result;
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> exit_time_mult_bm( double dim, double L){
  double ru1 = runif(1)[0];
  std::vector<double> X(dim), time(dim), Y((dim+2));
  int i = 0;
  X[i] = (ru1 < 0.5) ? -L : L;
  time[i] = exit_time_bm(L);
  double delta = time[i]; 
  int i_min = 0;
  for( i =1; i < dim; i++){
    ru1 = runif(1)[0];
    X[i] = (ru1 < 0.5) ? -L : L;
    time[i] = exit_time_bm(L);
    if(time[i] < delta){
      delta = time[i]; 
      i_min = i; 
    }
  }
  for(int j = 0; j < dim; j++){
    if( j==i_min ){
      Y[j] = X[j];
      continue;
    }else {
      Y[j] = bessel_bridge_pos(0,0,time[j],X[j],-L,L,delta);
    }
  }
  Y[dim] = delta; Y[dim+1] = i_min;
  return Y;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<double> exit_time_mult_bm_n(int n, double d, double L){
  std::vector<double> result(n);
  for(int i=0; i<n; i++){
    result[i] = exit_time_mult_bm(d,L)[d];
  }
  return result;
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
std::vector<std::vector<double>> single_segment_mult (double s, std::vector<double> W_s,
                                                      double dim, double L = 1){
  double dim_ske = 4*dim+3;
  std::vector<std::vector<double>> SKE(dim_ske);
  bool kill = false;
  double cur_time = s;
  double wait_time = 0, U_kill, prob, tau, U_tau,i_min, unif = 0;
  std::vector<double> W_tau(dim), lower(dim), upper(dim),W_tau_tau, cur_val = W_s;
  int time_size, seg_size;
  double Max = 0;
  while  (kill == false){ // (cur_time < 100000){// 
    U_tau = runif(1)[0];
    W_tau_tau = exit_time_mult_bm(dim,L);
    i_min = W_tau_tau[dim+1];
    tau = s + W_tau_tau[dim];
    for(int i =0; i < dim; i++){
      W_tau[i] = W_s[i] + W_tau_tau[i]; lower[i] = W_s[i] - L; 
      upper[i] = W_s[i] + L;
    }
    Max = M_mult(lower, upper);
    // Rcpp::Rcout << "max = " << Max << endl;
    // Rcpp::Rcout << "tau = " << tau << endl;
    SKE[0].push_back(s); SKE[1].push_back(tau); SKE[2].push_back(0);
    for(int k =0; k < dim; k++){
      SKE[3+k].push_back(W_s[k]);
      SKE[3+dim+k].push_back(W_tau[k]);
      SKE[3+2*dim+k].push_back(lower[k]);
      SKE[3+3*dim+k].push_back(upper[k]);
    }
    // Time.push_back(s); Pos.push_back(W_s); U_layer.push_back(W_s+L); L_layer.push_back(W_s-L); Tau.push_back(tau); W_Tau.push_back(W_tau); PTY.push_back(0);
    // Rcpp::Rcout << "First storage done!" << endl;
    while (cur_time < tau){
      Max = M_mult(lower, upper);
      // Rcpp::Rcout << "max = " << Max << endl;
      unif = runif(1)[0];
      wait_time = rexp(1,Max)[0];
      // Rcpp::Rcout << " wait time " << wait_time << endl;
      cur_time = cur_time + wait_time;
      if(cur_time > tau){
        cur_time = tau;
        break;
      }
      time_size = SKE[0].size()-1;
      for(int k =0; k < dim; k++){
        // Rcpp::Rcout << "SKE[0][time_size] = " << SKE[0][time_size] << " SKE[3+k][time_size] "<< SKE[3+k][time_size] << endl;
        // Rcpp::Rcout << "tau = " << tau << " W_tau[k] = " << W_tau[k] << "lower[k] = " << lower[k] << "upper[k] = " << upper[k] <<
        //  "cur_time = " << cur_time << endl;
        if (k == i_min){
          cur_val[k] = bessel_bridge_pos(SKE[0][time_size],SKE[3+k][time_size],tau,W_tau[k],lower[k],upper[k],cur_time);
        }else
          cur_val[k] = constr_brow_bridge(SKE[0][time_size],SKE[3+k][time_size],tau,W_tau[k],lower[k],upper[k],cur_time);
      }
      // cur_val = bessel_bridge_pos(Time[time_size],Pos[time_size],tau,W_tau,lower,upper,cur_time);
      U_kill = runif(1)[0];
      prob = PHI_mult(cur_val,lower,upper)/Max;
      if(prob > 1){
        Rcpp::Rcout << "Value exceeded by Maximum" << " Phi" << prob*Max <<
          " Max" << Max << " Cur_val" << cur_val[0] << endl;
        Rcpp::Rcout << "cur_time" << cur_time << "Tau" << tau << endl;
      }
      SKE[0].push_back(cur_time); SKE[1].push_back(tau); SKE[2].push_back(1);
      for(int k =0; k < dim; k++){
        SKE[3+k].push_back(cur_val[k]);
        SKE[3+dim+k].push_back(W_tau[k]);
        SKE[3+2*dim+k].push_back(lower[k]);
        SKE[3+3*dim+k].push_back(upper[k]);
      }
      // Time.push_back(cur_time); Pos.push_back(cur_val); U_layer.push_back(upper); L_layer.push_back(lower);Tau.push_back(tau); W_Tau.push_back(W_tau);PTY.push_back(1);
      if(U_kill < prob){
        kill = true;
        seg_size = SKE[0].size();
        SKE[2][(seg_size-1)] = 13;
        break;
      }
    }
    if(kill == false){
      SKE[0].push_back(tau); SKE[1].push_back(tau); SKE[2].push_back(2);
      for(int k =0; k < dim; k++){
        SKE[3+k].push_back(W_tau[k]);
        SKE[3+dim+k].push_back(W_tau[k]);
        SKE[3+2*dim+k].push_back(lower[k]);
        SKE[3+3*dim+k].push_back(upper[k]);
      }
      s = tau; W_s = W_tau;
    }
  }
  // rescale_object result;
  // result.pos = Pos; result.time = Time; result.u_layer = U_layer; result.l_layer = L_layer; result.TAU = Tau; result.W_TAU = W_Tau; result.pty = PTY;
  return SKE;
}


std::vector<std::vector<double>> rescale_layered_mult(double max_time, std::vector<double> W_s,
                                                      double dim, double L =1, double Lambda =0,
                                                      double prior_C = 0){
  double dim_ske = 4*dim+3;
  std::vector<std::vector<double>> segment(dim_ske), full_skeleton(dim_ske);
  double s =0,t,tau, cur_time = 0, seg_size, t_kill, reg_time;
  std::vector<double> reg_val(dim), W_t(dim), lower(dim),upper(dim), W_tau(dim);
  int clo_id;
  while (cur_time < max_time){
    segment = single_segment_mult(s,W_s,dim,L);
    for(int i =0; i < dim_ske; i++){
      full_skeleton[i].insert(full_skeleton[i].end(), segment[i].begin(),segment[i].end());
    }
    seg_size = segment[0].size()-1;
    t_kill = segment[0][seg_size];
    reg_time = runif(1, Lambda*t_kill, t_kill)[0];
    clo_id = search_elements(full_skeleton[0],reg_time);
    s = full_skeleton[0][clo_id]; t = full_skeleton[0][clo_id+1]; tau = full_skeleton[1][clo_id];
    for(int j=0; j < dim; j++){
      W_s[j] = full_skeleton[3+j][clo_id], W_t[j] = full_skeleton[3+j][clo_id+1];
      W_tau[j] = full_skeleton[3+dim+j][clo_id];
      lower[j] = full_skeleton[3+2*dim+j][clo_id];
      upper[j] = full_skeleton[3+3*dim+j][clo_id];
    }
    if(runif(1)[0] <= (1 - prior_C/(prior_C+t_kill))){
      for(int i=0; i<dim; i++){
        reg_val[i] = bessel_inter_point(s,W_s[i],t,W_t[i],tau,W_tau[i],lower[i],upper[i],reg_time);
      }
      full_skeleton[0].emplace(full_skeleton[0].begin()+clo_id+1,reg_time);
      full_skeleton[1].emplace(full_skeleton[1].begin()+clo_id+1,tau);
      full_skeleton[2].emplace(full_skeleton[2].begin()+clo_id+1,4);
      for(int j =0; j < 4; j++){
        switch(j){
        case 0 : for(int k_0=0; k_0 < dim; k_0++){
          full_skeleton[3+k_0].emplace(full_skeleton[3+k_0].begin()+clo_id+1,reg_val[k_0]);
        }break;
        case 1 : for(int k_1 =0; k_1 < dim; k_1++){
          full_skeleton[3+dim+k_1].emplace(full_skeleton[3+dim+k_1].begin()+clo_id+1,W_tau[k_1]);
        }break;
        case 2 : for(int k_2 =0; k_2 < dim; k_2++){
          full_skeleton[3+2*dim+k_2].emplace(full_skeleton[3+2*dim+k_2].begin()+clo_id+1,lower[k_2]);
        }break;
        case 3 : for(int k_3 =0; k_3 < dim; k_3++){
          full_skeleton[3+3*dim+k_3].emplace(full_skeleton[3+3*dim+k_3].begin()+clo_id+1,upper[k_3]);
        }break;
        }
      }
    }else{
      for(int i=0; i<dim; i++){
        reg_val[i] = rnorm(1,0,10)[0];
      }
    }
    s = t_kill;
    W_s = reg_val;
    cur_time = t_kill;
  }
  return full_skeleton;
}    


#include <string>
#include <sstream>

namespace patch
{
template < typename T > std::string to_string( const T& n )
{
  std::ostringstream stm ;
  stm << n ;
  return stm.str() ;
}
}  


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::DataFrame convert_to_dataframe(double max_time, std::vector<double> W_s,
                                     double dim, double L =1, double Lambda = 0,
                                     double prior_C = 0){
  std::vector<std::vector<double>> SKE = rescale_layered_mult(
    max_time,W_s,dim,L, Lambda, prior_C);
  Rcpp::DataFrame result(SKE);
  Rcpp::CharacterVector names(4*dim+3);
  names[0] = "t"; names[1] = "tau"; names[2] = "pty";
  for(int j =0; j < 4; j++){
    switch(j){
    case 0 : for(int k=0; k < dim; k++){
      names[3+k] = "x" + patch::to_string(k+1);
    }break;
    case 1 : for(int k =0; k < dim; k++){
      names[3+dim+k]= "W_tau" + patch::to_string(k+1);
    }break;
    case 2 : for(int k =0; k < dim; k++){
      names[3+2*dim+k] = "L" + patch::to_string(k+1);
    }break;
    case 3 : for(int k =0; k < dim; k++){
      names[3+3*dim+k] = "U" + patch::to_string(k+1);
    }break;
    }
  }
  result.attr("names") = names;
  return result;
}
