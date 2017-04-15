#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /*
   * Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  
  /*
   *  If size of estimations is 0, or the size of estimations and ground_truth
   *  do not match, return 0
   */
  if ( (estimations.size() != ground_truth.size()) ||
       (0 == estimations.size()) ) {
    return rmse;
  }
  
  /* 
   * Now calculate the rmse
   * rmse = sqrt( sum (X_est[i] - X_true[i])^2 / n)
   */
  for (int i = 0; i < estimations.size(); i++) {
    // R = X_est[i] - X_true[i]
    VectorXd r = estimations[i] - ground_truth[i];
    
    // R^2
    r = r.array() * r.array();
    
    // sum
    rmse = rmse + r;
  }
  
  // divide by n (i.e. estimation.size())
  rmse = rmse / estimations.size();
  
  // Now take the sqrt
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /*
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3, 4);
  
  // declare state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  double px_2 = px * px;
  double py_2 = py * py;
  double vx_py = vx * py;
  double vy_px = vy * px;
  double px_py_2 = px_2 + py_2;
  double px_py_sq_2 = sqrt(px_py_2);
  
  /*
   * If denominator is 0, throw an error
   */
  if (fabs(px_py_2) < 0.0001) {
    std::cout << "Jacbian Error - Divide by zero\n";
    return Hj;
  }
  
  /*
   * Setup the Jacbian Matrix
   */
  Hj << px / px_py_sq_2, py / px_py_sq_2, 0, 0,
        -py / px_py_2, px / px_py_2, 0, 0,
        py * (vx_py - vy_px) / pow(px_py_2, 1.5), px * (vy_px - vx_py) / pow(px_py_2, 1.5), px / px_py_sq_2, py / px_py_sq_2;

  // return
  return Hj;
}
