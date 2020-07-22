#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
mat ortho_mat(const mat &input);

mat ortho_mat(const mat &input){
  int n = input.n_rows;
  int p = input.n_cols;
  vec intercept(n, fill::ones);
  mat z(n,p,fill::ones);
  for (int i=0; i<p; i++){
    if (i==0){
      vec tmp = input.col(0)-dot(input.col(0),intercept)/dot(intercept,intercept)*intercept;
      z.col(0) = tmp;
    }
    else{
      vec input_tmp = input.col(i);
      vec tmp = input_tmp -dot(input_tmp,intercept)/dot(intercept,intercept)*intercept;
      for (int j=0; j<i; j++){
        tmp = tmp - dot(input_tmp,z.col(j))/dot(z.col(j),z.col(j))*z.col(j);
      }
      z.col(i) = tmp;
    }
  }
  return z;
}