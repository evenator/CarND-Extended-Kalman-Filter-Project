#include "FusionEKF.h"
#include <cmath>
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  x_ = VectorXd::Zero(4);
  P_ = MatrixXd::Identity(4, 4) * initial_variance_;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      x_ = HInvRadar(measurement_pack.raw_measurements_);
      MatrixXd HInvj = HInvjRadar(x_);
      P_ = HInvj * R_radar_ * HInvj.transpose();
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = measurement_pack.raw_measurements_(0);
      x_(1) = measurement_pack.raw_measurements_(1);
      P_(0, 0) = R_laser_(0, 0);
      P_(1, 1) = R_laser_(1, 1);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    // print the output
    cout << "x:" << endl << x_ << endl;
    cout << "P:" << endl << P_ << endl;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  Predict(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    VectorXd y = measurement_pack.raw_measurements_ - HRadar(x_);
    // Coerce phi to (-pi, pi)
    y(1) = atan2(sin(y(1)), cos(y(1)));
    Update(y, HjRadar(x_), R_radar_);
  } else {
    // Laser updates
    VectorXd y = measurement_pack.raw_measurements_ - H_laser_ * x_;
    Update(y, H_laser_, R_laser_);
  }

  previous_timestamp_ = measurement_pack.timestamp_;
}

MatrixXd FusionEKF::CalculateF(double dt) {
  assert(isfinite(dt));
  MatrixXd F = MatrixXd::Identity(4, 4);
  F(0, 2) = dt;
  F(1, 3) = dt;
  return F;
}

Eigen::VectorXd FusionEKF::HRadar(const Eigen::VectorXd &x_state) {
  assert(x_state.allFinite());
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  double p2 = px * px + py * py;
  if (p2 < 0.000001) {
    double v2 = vx * vx + vy * vy;
    // In the all-zero case, the z measurement is not defined by the x state
    if (v2 < 0.000001) {
      return VectorXd::Zero(3);
    }
    // Since v is non-zero, we can use it to determine the heading and
    // r_dot
    double v = sqrt(v2);
    return (VectorXd(3) << 0,
                           atan2(vy, vx),
                           v).finished();
  }
  double p = sqrt(p2);
  return (VectorXd(3) << p,
                         atan2(py, px),
                         (px * vx + py * vy) / p).finished();
}

Eigen::VectorXd FusionEKF::HInvRadar(const Eigen::VectorXd &z_measurement) {
  assert(z_measurement.allFinite());
  double r = z_measurement(0);
  double phi = z_measurement(1);
  double r_dot = z_measurement(2);
  double c = cos(phi);
  double s = sin(phi);
  return (VectorXd(4) << r * c,
                         r * s,
                         r_dot * c,
                         r_dot * s).finished();
}

Eigen::MatrixXd FusionEKF::HInvjRadar(const Eigen::VectorXd &z_measurement) {
  assert(z_measurement.allFinite());
  double r = z_measurement(0);
  double phi = z_measurement(1);
  double r_dot = z_measurement(2);
  double s = sin(phi);
  double c = sin(phi);
  return (MatrixXd(4,3) << c, -r * s, 0,
                           s, r * c, 0,
                           0, -r_dot * s, c,
                           0, r_dot * c, s).finished();
}

MatrixXd FusionEKF::HjRadar(const VectorXd &x_state) {
  assert(x_state.allFinite());
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  double p2 = px * px + py * py;
  // Handle all-zeros case.
  if (p2 < 0.000001) {
    double v2 = vx * vx + vy * vy;
    // In the all-zero case, the z measurement is not defined by the x state
    if (v2 < 0.000001) {
      return MatrixXd::Zero(3, 4);
    }
    // Since v is non-zero, we can use it to determine the heading and
    // r_dot
    double v = sqrt(v);
    return (MatrixXd(3, 4) << 0, 0, 0, 0,
                              0, 0, -vy / v2, vx / v2,
                              0, 0, 1 / v, 1 / v).finished();
  }
  double p = sqrt(p2);
  double p3 = p2 * p;
  MatrixXd Hj(3, 4);
  Hj  << px / p, py / p, 0, 0,
         -py / p2, px / p2, 0, 0,
         py * (vx * py - vy * px) / p3, px * (px * vy - py * vx) / p3, px / p, py / p;
  return Hj;
}

void FusionEKF::Predict(double dt) {
  // TODO: Preallocate F, Q, for speed?
  // state transition matrix
  Eigen::MatrixXd F = CalculateF(dt);

  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q(dt);
}

MatrixXd FusionEKF::Q(double dt) {
  assert(isfinite(dt));
  double noise_ax = 9;
  double noise_ay = 9;
  double dt2 = dt * dt;
  double dt3 = dt2 * dt;
  double dt4 = dt3 * dt;
  MatrixXd Q(4, 4);
  Q << dt4 / 4 * noise_ax, 0, dt3 / 2 * noise_ax, 0,
       0, dt4 / 4 * noise_ay, 0, dt3 / 2 * noise_ay,
       dt3 / 2 * noise_ax, 0, dt2 * noise_ax, 0,
       0, dt3 / 2 * noise_ay, 0, dt2 * noise_ay;
  return Q;
}

void FusionEKF::Update(const VectorXd &y, const MatrixXd &H,
                       const MatrixXd &R) {
  assert(y.allFinite());
  assert(H.allFinite());
  assert(R.allFinite());
  // TODO: Preallocate S, K, for speed?
  MatrixXd S = H * P_ * H.transpose() + R;
  // TODO: It's more efficient to use Eigen's solver than an explicit inverse
  MatrixXd K = P_ * H.transpose() * S.inverse();
  x_ = x_ + K * y;
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H) * P_;
  cout << endl
       << "y:" << endl << y << endl
       << "H:" << endl << H << endl
       << "R:" << endl << R << endl
       << "S:" << endl << S << endl
       << "K:" << endl << K << endl
       << "KH:" << endl << K*H << endl
       << "X:" << endl << x_ << endl
       << "P:" << endl << P_ << endl
       << endl;
  assert(x_.allFinite());
  assert(P_.allFinite());
}
