/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, you can obtain one at http://mozilla.org/MPL/2.0/.
* ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * Softshrink(x) =
 * x - lambda, if x > lambda
 * x + lambda, if x < lambda
 * 0, otherwise
 *
 * @property lambda defines the lambda value
 */
class Softshrink(val lambda: Double = 0.5) : ScalarActivationFunction() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check if lambda is positive
   */
  init {
    require(this.lambda > 0.0)
  }

  /**
   * Calculate the Softplus function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = when {
    x > this.lambda -> x - this.lambda
    x < -this.lambda -> x + this.lambda
    else -> 0.0
  }

  /**
   * Optimized derivative of the Softplus function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the Softplus derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = when {
    fx + this.lambda > this.lambda -> 1.0
    fx - this.lambda < -this.lambda -> 1.0
    else -> 0.0
  }
}
