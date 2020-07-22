#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>

using namespace Rcpp;
using namespace std;
using namespace arma;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
mat weight_calc(const mat &X, const mat &class_mean, const cube &class_var, const vec &proportion);

// [[Rcpp::export]]
mat proportion_calc(const mat &weight);

// [[Rcpp::export]]
mat mean_calc(const mat &weight, const mat &X, const mat &proportion);

// [[Rcpp::export]]
cube variance_calc(const mat &weight, const mat &X, const mat &class_mean);

/*E-Step*/
mat weight_calc(const mat &X, const mat &class_mean, const cube &class_var, const vec &proportion){
  int n = X.n_rows;
  int p = X.n_cols;
  int k = class_mean.n_cols;
  cube inv_cov(p,p,k);
  mat weights(n,k);
  for (int i=0; i<k; i++){
    mat slice_inv =inv_sympd(class_var.slice(i));
    inv_cov.slice(i) = slice_inv;
  }
  for (int i=0; i<n; i++){
    vec weights_i(k);
    mat x_i = X.row(i);
    for (int j=0; j<k; j++){
      weights_i(j) = mean(mean(proportion(j)*sqrt(det(inv_cov.slice(j)))*exp(-(0.5)*(x_i-class_mean.col(j).t())*inv_cov.slice(j)*((x_i-class_mean.col(j).t()).t()))));
    }
    weights.row(i) = (weights_i/sum(weights_i)).t();
  }
  return weights;
}


/*M-Step for proportion*/
mat proportion_calc(const mat &weight){
  mat prop = mean(weight);
  return prop;
}

/*M-Step for mean*/
mat mean_calc(const mat &weight, const mat &X, const mat &proportion){
  int n = X.n_rows;
  mat class_mean = diagmat((1/proportion))*weight.t()*X/n;
  return class_mean.t();
}

/*M-Step for variance
cube variance_calc(const mat &weight, const mat &X, const mat &class_mean, const mat &proportion){
  int n = X.n_rows;
  int k = weight.n_cols;
  int p = X.n_cols;
  cube class_var(p,p,k);
  for (int i=0; i<k; i++){
    mat rep_mean = zeros<mat>(n,p);
    rep_mean.each_row() += (class_mean.col(i).t());
    class_var.slice(i) = (X-rep_mean).t()*diagmat(weight.col(i))*(X-rep_mean)/(n*proportion(0,i));
  }
  return class_var;
}
 */
  
