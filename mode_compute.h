#pragma once
#include <RcppArmadillo.h>
#include <math.h>
#include <iostream>

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
int vec_mode(const ivec &input);

int vec_mode(const ivec &input){
  int number = input(0);
  int mode = number;
  int count = 1;
  int countMode = 1;
  int length = input.n_elem;
  for (int i=1; i< length; i++){
    if (input(i) == number) { 
      count++;
    }
    else{ 
      if (count > countMode) {
        countMode = count; 
        mode = number;
      }
      count = 1; 
      number = input(i);
    }
  }
  if (count > countMode) {
    countMode = count; 
    mode = number;
  }
  return mode;
}

