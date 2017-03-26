#include "tools.h"
#include <iostream>

using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth) {
  assert(estimations.size() == ground_truth.size());
  assert(estimations.size() != 0);
  ArrayXd sum_err2 = ArrayXd::Zero(estimations.front().size());
  for (size_t i = 0; i < estimations.size(); ++i) {
    assert(estimations[i].size() == ground_truth[i].size());
    ArrayXd err = (estimations[i] - ground_truth[i]).array();
    sum_err2 += err * err;
  }
  return (sum_err2 / estimations.size()).sqrt().matrix();
}
