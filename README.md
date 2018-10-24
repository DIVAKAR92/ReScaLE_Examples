# How to run the ReScaLE algorithm?

## Steps:
1. Write RCpp function for the hazard rate i.e. kappa and wrap it inside PHI_mult(). 
2. Write RCpp function for the upper bound of kappa and wrap it inside M_mult().
3. Rcpp::sourceCpp('file_name.cpp') in the R environment.
4. Make a call to convert_to_dataframe() to create the ReScaLE skeleton.
5. Then specify a time mesh. 
6. Make a call to skeleton_at_given_mesh() with relevant arguments. 

## Example:
See menarche_rescale_runs.R for a logistic regression model on the Menarche data.


