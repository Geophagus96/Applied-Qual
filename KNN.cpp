#include <RcppArmadillo.h>
#include <math.h>
#include <mode_compute.h>
#include <iostream>

using namespace Rcpp;
using namespace std;
using namespace arma;

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
ivec class_test_KNN(const mat &train, const mat &test, const ivec &label, const int &nneigh){
  int n_train = train.n_rows;
  int n_test = test.n_rows;
  ivec test_class(n_test);
  for (int i=0; i<n_test; i++){
    mat test_sample = test.row(i);
    mat test_mat = repmat(test_sample,n_train,1);
    vec dist = sum(pow(train-test_mat,2),1);
    int sample_class;
    if(nneigh == 1){
      uword test_index = dist.index_min();
      sample_class = label(test_index);
    }
    else{
      uvec sorted_index = sort_index(dist);
      uvec test_index = sorted_index.head(nneigh);
      ivec neighbor_class = label.elem(test_index);
      ivec sort_neighbor_class = sort(neighbor_class);
      sample_class = vec_mode(sort_neighbor_class);
    }
    test_class(i) = sample_class;
  }
  return test_class;
}

// [[Rcpp::export]]
ivec inter_class_KNN(const mat &train, const ivec &label, const int &nneigh){
  int n_train = train.n_rows;
  ivec train_class(n_train);
  for (int i=0; i<n_train; i++){
    mat train_sample = train.row(i);
    mat train_mat = repmat(train_sample,n_train,1);
    vec dist = sum(pow(train-train_mat,2),1);
    uvec sorted_index = sort_index(dist);
    uvec top_sorted_index = sorted_index.head((nneigh+1));
    uvec neigh_index = top_sorted_index.tail(nneigh);
    int sample_class;
    if (nneigh == 1){
      uword train_index = neigh_index(0);
      sample_class = label(train_index);
    }
    else{
      ivec neighbor_class = label.elem(neigh_index);
      ivec sort_neighbor_class = sort(neighbor_class);
      sample_class = vec_mode(sort_neighbor_class);
    }
    train_class(i) = sample_class;
  }
  return train_class;
}


