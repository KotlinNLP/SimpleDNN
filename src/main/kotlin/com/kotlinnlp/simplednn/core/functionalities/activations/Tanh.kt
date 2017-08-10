/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * The hyperbolic tangent tanh(x) activation function is an S-shaped function,
 * transforming the values x into the range [−1, 1].
 */
class Tanh : ScalarActivationFunction() {

  /**
   * Apply the activation function to [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = Math.tanh(x)

  /**
   * Optimized derivative of the Tanh function, calculated in x
   *
   * @param fx input (WARNING: it must be f(x) for optimization)
   *
   * @return the Tanh derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = 1.0 - Math.pow(fx, 2.0)

}
