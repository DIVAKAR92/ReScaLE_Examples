#include <Rcpp.h> 
#include <stdlib.h>
#include <vector>
#include <algorithm>
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


// RESCALE CODE ==========================================

// the structure which will hold the positions at different times for the Brownian 
// motion. 
struct rescale_object {
  std::vector<double> time; 
  std::vector<double> pos; 
  std::vector<double> pty;
};

// Function returns the simulated Brownian bridge positions at some intermediate points
// of a given left and right positions of a Brownian motion. 
std::vector<double> two_point_bb(
    std::vector<double> times, // already known times as a vector of 2 elements 
    std::vector<double> pos,   // already known positions  as a vector of 2 elements
    std::vector<double> req_time // intermediate times at which position is sought
){
  
  int N = req_time.size(); // size of required positions
  std::vector<double>  new_times(N + 2);  // newly created times as a vector of (N+2) elements
  std::vector<double>  new_pos(N + 2); // Newly created vector of positions
  std::vector<double> result(N); // stores the positions at required times. 
  
  int i = 0;
  new_times[i] = times[i]; new_pos[i] = pos[i]; // initialisations 
  new_times[N+1] = times[1]; new_pos[N+1] = pos[1]; 
  
  // stores the mean and sd for the normal distribution for caculation of BB
  double mean, sd;
  
  if (N != 0) 
  {
    
    // loop runs through N times and evaluates the positions at required times 
    // using the definition of Brownian bridge.
    for (i = 1; i < (N+1); i++) {
      // loop that fills the required times at newly created time vector 
      new_times[i] = req_time[i-1];
      
      // calculate the mean and sd of normal distribution to be used to simulate BB
      mean = (new_pos[N+1]*(new_times[i] - new_times[i-1]) + 
        new_pos[i-1]*(new_times[N+1] - new_times[i]))/
      (new_times[N+1] - new_times[i-1]);
      sd = sqrt((new_times[i] - new_times[i-1])*(new_times[N+1] - 
        new_times[i])/(new_times[N+1] - new_times[i-1]));
      // simulated BB position using a normal distribution
      new_pos[i] = rnorm(1, mean, sd)[0];
      // store the simulated position in the result/output
      result[i-1] = new_pos[i];
    }
  }
  return(result);
}



// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
std::vector<double> posterior(std::vector<double> x){
  double prod;
  int Nx = x.size();
  std::vector<double> result(Nx);
  for (int j=0; j < Nx; j++){
    prod = 1;
    for(int i=0; i <N; i++){
      prod = prod*1/(1+pow((data[i]-x[j]),2));
    }
    prod = prod * 1/(1+x[j]*x[j]);
    result[j] = 1/0.06281079*prod; 
  }
  return result;
}  

// Function returns one segment/skeleton of a killed Brownian motion for a given 
// startting time and starting position.
rescale_object segments (double start_time, double start_pos, double M )
{
  std::vector<double> times; // stores the poisson event times 
  std::vector<double> pos;  //  stores the positions visited by the Brownian motion
  std::vector<double> PTY; 
  times.push_back(start_time); // store the starting time 
  pos.push_back(start_pos); // store the initial poistion in the position vector.
  PTY.push_back(0); 
  int kill = 0; // a check to evaluate if kill occurs or not
  //  numeric values to store the current time and current positions.
  double cur_time = start_time; double cur_val = start_pos;
  double phi_val;
  
  // we loop until kill happens for a Brownian motion. 
  while (kill == 0){
    double wait_time = rexp(1, M)[0]; // simulate the waiting time for the poisson event 
    cur_time += wait_time; // get the actual poisson event times. 
    double BM = rnorm(1, cur_val, sqrt(wait_time))[0]; // simulate the Brownian motion at current time.
    phi_val = kappa(BM);
    if (runif(1)[0] < phi_val/M) // checks if kill occurs
    {
      kill = 1; // if kill occurs then change the kill value to 1.
      times.push_back(cur_time);  // store the kill time in the vector.
      pos.push_back(BM);  // store the kill poistion at the 
      PTY.push_back(13); 
    }
    else
    {
      kill = 0; // if the kill does not occur then kill is still 0.
      cur_val = BM; // in this case set the current value as the simulated Brownian motion position.
      times.push_back(cur_time); // store the current time in the vector 
      pos.push_back(BM);  // store the simulated position in the vector.
      PTY.push_back(1); 
    } 
  }
  rescale_object skeleton; // final result as the rescale_object type.
  skeleton.time = times; // store the poisson event times in the skeleton
  skeleton.pos  = pos;  // store the positions visited in the skeleton.
  skeleton.pty  = PTY; 
  return skeleton;
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


// Function returns the all the positions visited by a killed Brownian motion until some
// maximum time for a given starting poistion. The segments are acreated according to the 
// RESCALE methodology wherein once the brownian motion is killed then it is regenrated 
// by simulating from the states visited by it. 

rescale_object test_segments (double max_time, double start_pos, 
                              double M,double Lambda =0,
                              double prior_C = 0){
  rescale_object skeleton;
  std::vector<double> full_time, full_pos, full_pty;
  std::vector<double> seg_time, seg_pos, seg_pty;
  std::vector<double> reg_val(1);
  std::vector<double> reg_time(1);
  int N, clo_time;
  double t_kill, cur_time = 0, start_time = 0;
  std::vector<double> bb_time(2), bb_pos(2);
  std::vector<double>::iterator it1, it2, it3;
  int size = 0; 
  
  while(cur_time < max_time){
    skeleton = segments(start_time, start_pos, M);
    seg_time = skeleton.time;
    seg_pos = skeleton.pos;
    seg_pty = skeleton.pty;
    full_time.insert(full_time.end(),seg_time.begin(),seg_time.end());
    full_pos.insert(full_pos.end(),seg_pos.begin(),seg_pos.end());
    full_pty.insert(full_pty.end(),seg_pty.begin(),seg_pty.end());
    N = seg_time.size();
    t_kill = seg_time[N-1];
    reg_time[0] = runif(1, Lambda*t_kill, t_kill)[0];
    
    clo_time = index_last_less_ele(full_time,reg_time[0]);
    bb_time[0] = full_time[clo_time]; bb_time[1] = full_time[clo_time+1];
    bb_pos[0] = full_pos[clo_time]; bb_pos[1] = full_pos[clo_time+1];
    if(runif(1)[0] <= (1 - prior_C/(prior_C+t_kill))){
      reg_val = two_point_bb(bb_time,bb_pos,reg_time);
    }else{
      reg_val[0] = runif(1,-10.0,10.0)[0];
    }
    
    it1 = full_time.begin(); it2 = full_pos.begin(); it3 = full_pty.begin();
    full_time.emplace(it1+clo_time+1,reg_time[0]);
    full_pos.emplace(it2+clo_time+1,reg_val[0]);
    full_pty.emplace(it3+clo_time+1,2);
    
    start_time = t_kill;
    start_pos = reg_val[0];
    cur_time = t_kill;
    size = size + seg_time.size();
  }
  rescale_object result;
  result.time = full_time; result.pos = full_pos; result.pty = full_pty;
  return result;
} 

//=======================================================================================
// Function to restrict the outputs generated by test_segments to maximum time

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]] 
Rcpp::DataFrame rescale_skeleton(double max_time, double start_pos, double M, double Lambda =0,
                                 double prior_C = 0){
  
  // store the skeleton generated by the test_skeleton in temp_skeleton
  rescale_object temp_skeleton = test_segments(max_time, start_pos, M, Lambda, prior_C);
  // extract the event times and positions
  std::vector<double> full_time = temp_skeleton.time;
  std::vector<double> full_pos  = temp_skeleton.pos;
  std::vector<double> full_pty  = temp_skeleton.pty;
  // Store the last element in the sorted vector of full times which is less than regenerated time.
  int index = index_last_less_ele(full_time,max_time);
  std::vector<double> bb_time(2), bb_pos(2); // stores the nearest times and its positions.
  // store the two closest times to regenerated time for the purpose of calculating the brownian bridge value
  bb_time[0] = full_time[index]; bb_time[1] = full_time[index+1];
  // Store the corresponding positions of two closest times. 
  bb_pos[0] = full_pos[index]; bb_pos[1] = full_pos[index+1];
  std::vector<double> reg_val(1), m_time(1);
  m_time[0] = max_time;
  // Brownian brdige value is the Brownian bridge calculated at max_time.
  reg_val = two_point_bb (bb_time,bb_pos,m_time);
  int N = full_time.size()-index;
  for (int i = 0; i < N; i++ ){
    full_time.pop_back();
    full_pos.pop_back();
    full_pty.pop_back();
  }
  full_time.push_back(max_time);
  full_pos.push_back(reg_val[0]);
  full_pty.push_back(1);
  // rescale_object result;
  // result.time = full_time;
  // result.pos = full_pos;
  // return result;
  return(Rcpp::DataFrame::create(
      _["t"]= full_time, 
      _["x"] = full_pos,
      _["pty"] = full_pty));
}





