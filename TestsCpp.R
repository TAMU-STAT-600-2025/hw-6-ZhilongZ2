
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)
library(testthat)

# Source your C++ funcitons
sourceCpp("LassoInC.cpp")

# Source your LASSO functions from HW4 (make sure to move the corresponding .R file in the current project folder)
source("D:/25 Fall TAMU/STAT 600/hw-4-ZhilongZ2/LassoFunctions.R")

# Do at least 2 tests for soft-thresholding function below. You are checking output agreements on at least 2 separate inputs
#################################################
test_that("soft_c matches R soft() on simple scalar inputs", {
  expect_equal(soft_c(3, 1),  soft(3, 1))
  expect_equal(soft_c(-2, 0.5), soft(-2, 0.5))
})

test_that("soft_c returns 0 when |a| <= lambda", {
  expect_equal(soft_c(0.3, 1),  soft(0.3, 1))
  expect_equal(soft_c(-0.8, 1), soft(-0.8, 1))
})


# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################
test_that("lasso_c matches R lasso() on simple numeric inputs", {
  # Test 1
  X1 <- matrix(c(1, 2, 3, 4), nrow = 2)
  Y1 <- c(1, 2)
  beta1 <- c(0.5, -0.2)
  lambda1 <- 0.1
  expect_equal(lasso_c(X1, Y1, beta1, lambda1),
               lasso(X1, Y1, beta1, lambda1))
  
  # Test 2
  X2 <- matrix(rnorm(12), nrow = 3)
  Y2 <- rnorm(3)
  beta2 <- rnorm(4)
  lambda2 <- 0.5
  expect_equal(lasso_c(X2, Y2, beta2, lambda2),
               lasso(X2, Y2, beta2, lambda2))
})

# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################
test_that("fitLASSOstandardized_c matches R fitLASSOstandardized on small inputs", {
# Test 1
X1 <- scale(matrix(c(1, 2, 3, 4, 5, 6), nrow = 3), center = TRUE, scale = TRUE)
Y1 <- scale(c(1, 2, 3), center = TRUE, scale = FALSE)
lambda1 <- 0.1
beta_start1 <- numeric(ncol(X1))

beta_r1 <- fitLASSOstandardized(X1, Y1, lambda1, beta_start = beta_start1)$beta
beta_c1 <- fitLASSOstandardized_c(X1, Y1, lambda1, beta_start1)
expect_equal(as.numeric(beta_c1), as.numeric(beta_r1), tolerance = 1e-6)

# Test 2
X2 <- scale(matrix(rnorm(15), nrow = 5), center = TRUE, scale = TRUE)
Y2 <- scale(rnorm(5), center = TRUE, scale = FALSE)
lambda2 <- 0.2
beta_start2 <- numeric(ncol(X2))

beta_r2 <- fitLASSOstandardized(X2, Y2, lambda2, beta_start = beta_start2)$beta
beta_c2 <- fitLASSOstandardized_c(X2, Y2, lambda2, beta_start2)
expect_equal(as.numeric(beta_c2), as.numeric(beta_r2), tolerance = 1e-6)
})

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################

# Do microbenchmark on fitLASSOstandardized_seq vs fitLASSOstandardized_seq_c
######################################################################

# Tests on riboflavin data
##########################
require(hdi) # this should install hdi package if you don't have it already; otherwise library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene erpression

# Make sure riboflavin$x is treated as matrix later in the code for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]

# Standardize the data
out <- standardizeXY(riboflavin$x, riboflavin$y)

# This is just to create lambda_seq, can be done faster, but this is simpler
outl <- fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, n_lambda = 30)

# The code below should assess your speed improvement on riboflavin data
microbenchmark(
  fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq),
  fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq),
  times = 10
)
