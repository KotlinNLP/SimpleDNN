/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * Softplus(x) = 1 / beta ∗ log(1 + exp(beta ∗ x))
 *
 * @property beta defines the decreasing exponential rate for the negative values. Defaults to 1.0
 * @property threshold defines a threshold, for numerical stability (if x > threshold, f(x) = x)
 */
class Softplus(val beta: Double = 1.0, val threshold: Double = 20.0) : ScalarActivationFunction() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check if beta is positive
   */
  init {
    require(this.beta > 0.0)
  }

  /**
   * Calculate the Softplus function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = if (x < this.threshold) 1.0 / this.beta * Math.log(1.0 + Math.exp(x * this.beta))
  else x

  /**
   * Optimized derivative of the Softplus function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the Softplus derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = if (fx < this.threshold) (Math.exp(this.beta * fx) - 1.0) /
      (Math.exp(this.beta * fx)) else 1.0

}
