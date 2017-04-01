#ifndef FusionEKF_H_
#define FusionEKF_H_

#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"

class FusionEKF {
 public:
  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

 private:
  // check whether the tracking toolbox was initiallized or not (first
  // measurement)
  bool is_initialized_;

  const double initial_variance_ = 1e3;

  // previous timestamp
  long previous_timestamp_;

  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;

  /**
   * The non-linear measurement function for the radar
   * @param x_state The object state to convert to measurement space
   * @return The position, speed, bearing, and range rate
   */
  Eigen::VectorXd HRadar(const Eigen::VectorXd &x_state);

  /**
   * The inverse non-linear measurement function for the radar
   * @param z_measurement The measurement vector to convert to state space
   * @return The x position, y position, x velocity, and y velocity
   */
  Eigen::VectorXd HInvRadar(const Eigen::VectorXd &z_measurement);

  /**
   * Calculate the state transition matrix as a function of delta t
   * @param dt The delta time of the state transition (seconds)
   * @return The F matrix
   */
  Eigen::MatrixXd CalculateF(double dt);

  /**
   * The linearized (Jacobian) H matrix for the ladar, as a function of the
   * state
   * @param x_state The object state to convert to measurement space
   * @return The H matrix
   */
  Eigen::MatrixXd HjRadar(const Eigen::VectorXd &x_state);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param dt Time between k and k+1 in s
   */
  void Predict(double dt);

  /**
   * Calculate the process noise matrix as a function of delta t
   * @param dt The delta time of the state transition (seconds)
   * @return The Q matrix
   */
  Eigen::MatrixXd Q(double dt);

  /**
   * Generic update that performs the similar steps of EKF and KF updates
   * @param y The measurement after applying H
   * @param H or Hj as appropriate
   * @param R for the measurement
   */
  void Update(const Eigen::VectorXd &y, const Eigen::MatrixXd &H,
              const Eigen::MatrixXd &R);
};

#endif /* FusionEKF_H_ */
