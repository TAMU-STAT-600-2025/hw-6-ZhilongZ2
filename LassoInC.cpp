#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Soft-thresholding function, returns scalar
// [[Rcpp::export]]
double soft_c(double a, double lambda){
  // assumes lambda >= 0
  if (a >  lambda) return a - lambda;
  if (a < -lambda) return a + lambda;
  return 0.0;
}

// Lasso objective function, returns scalar
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& beta, double lambda){
  // Your function code goes here
  int n = Xtilde.n_rows;
  arma::colvec r = Ytilde - Xtilde * beta;
  double loss = arma::dot(r, r) / (2.0 * n);
  double pen  = lambda * arma::accu(arma::abs(beta));
  return loss + pen;
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, 
                                    double lambda, const arma::colvec& beta_start, double eps = 0.001){
  // Your function code goes here
  const arma::uword n = Xtilde.n_rows;
  const arma::uword p = Xtilde.n_cols;
  
  arma::colvec beta = beta_start;                 
  arma::colvec r = Ytilde - Xtilde * beta;        
  
  double f_prev = arma::dot(r, r) / (2.0 * n) + lambda * arma::accu(arma::abs(beta));
  
  const int max_iter = 10000;
  for (int k = 0; k < max_iter; ++k) {
    // refresh residuals at start of sweep
    r = Ytilde - Xtilde * beta;

    for (arma::uword j = 0; j < p; ++j) {
      const arma::colvec xj = Xtilde.col(j);
      const double zj = beta(j) + arma::dot(xj, r) / static_cast<double>(n);
      const double beta_new_j = soft_c(zj, lambda);

      const double delta = beta(j) - beta_new_j;
      if (delta != 0.0) r += xj * delta;          // coordinate-wise residual update
      beta(j) = beta_new_j;
    }
    
    const double f_curr =
      arma::dot(r, r) / (2.0 * n) + lambda * arma::accu(arma::abs(beta));
    
    if ((f_prev - f_curr) < eps) return beta;     // stop at first time below eps
    f_prev = f_curr;
  }
  
  return beta;
}

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq, double eps = 0.001){
  // Your function code goes here
  const arma::uword p = Xtilde.n_cols;
  const arma::uword L = lambda_seq.n_elem;
  
  arma::mat beta_mat(p, L);
  arma::colvec beta_start(p, arma::fill::zeros);  // warm start: zeros for the largest lambda

  for (arma::uword i = 0; i < L; ++i) {
    const double lam = lambda_seq(i);
    arma::colvec beta = fitLASSOstandardized_c(Xtilde, Ytilde, lam, beta_start, eps);
    beta_mat.col(i) = beta;
    beta_start = beta;  // warm start for the next (smaller) lambda
  }

  return beta_mat;
}