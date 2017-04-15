#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// define PI constant value
const float PI = 3.1415926;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /*
   * predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update_(const VectorXd &y) {
  //VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // Now calculate the new estimate
  x_ = x_ +  (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
   * update the state by using Extended Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  Update_(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  /*
   * [rho, theta, drho] = [ sqrt(px^2 + py^2), arctan(py/px), 
   *                        (px*vx+py*vy)/sqrt(px^2 + py^2) ]
   */
  VectorXd z_pred(3);
  float px, py, vx, vy;
  px = x_(0); py = x_(1); vx = x_(2); vy = x_(3);

  float rho   = sqrt(pow(px, 2) + pow(py, 2));
  float theta = atan2(py, px);
  float drho  = 0;
  
  // check division by zero
  if (rho < 0.0001) {
    drho = (px * vx + py * vy) / 0.0001;
  }
  else {
    drho = (px * vx + py * vy) / rho;
  }
  
  z_pred << rho, theta, drho;
  
  VectorXd y = z - z_pred;
  
  /*
   * atan2() returns values between -pi and pi. When calculating phi in 
   * y = z - h(x) for radar measurements, the resulting angle phi in the 
   * y vector should be adjusted so that it is between -pi and pi. 
   * The Kalman filter is expecting small angle values between the range 
   * -pi and pi. HINT: when working in radians you can add 2π or subtract 
   * 2π until the angle is within the desired range.
   */
  while (1) {
    if (y(1) > PI) {
      y(1) = y(1) - (2 * PI);
    }
    else if (y(1) < -PI) {
      y(1) = y(1) + (2 * PI);
    }
    else {
      break;
    }
  }
  
  Update_(y);
}
