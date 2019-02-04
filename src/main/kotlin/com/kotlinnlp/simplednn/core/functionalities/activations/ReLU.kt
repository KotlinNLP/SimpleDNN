/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * The Rectifier activation function clips each value x < 0 at 0,
 * transforming each value x into the range [0,∞]
 *
 * References
 * [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
 */
class ReLU : ScalarActivationFunction() {

  /**
   * Calculate the ReLU function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = if (x <= 0.0) 0.0 else x

  /**
   * Optimized derivative of the ReLU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the ReLU derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = if (fx > 0.0) 1.0 else 0.0

}
