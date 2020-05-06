/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, you can obtain one at http://mozilla.org/MPL/2.0/.
* ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * LeakyRELU(x) = (max(0, x) + min(0, slope * x))
 *
 * @property slope defines the decreasing rate for the negative values. Defaults to 0.01
 *
 */
class LeakyRELU(val slope: Double = 0.01) : ScalarActivationFunction {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check if slope is positive
   */
  init {
    require(this.slope > 0.0)
  }

  /**
   * Calculate the LeakyReLU function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = if (x <= 0.0) this.slope * x else x

  /**
   * Optimized derivative of the LeakyReLU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the LeakyReLU derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = if (fx > 0.0) 1.0 else this.slope
}
