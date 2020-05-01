/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * CeLU(x) = max(0, x) + min(0, α ∗ (exp(x / α) − 1)))
 *
 * References
 * [Jonathan T. Barron, 2017, Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483)
 *
 * @property alpha defines the decreasing exponential rate for the negative values. Defaults to 1.0
 *           Using default value, the function is identical to ELU.
 *
 */
class CeLU(val alpha: Double = 1.0) : ScalarActivationFunction() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check if alpha is positive
   */
  init {
    require(this.alpha > 0.0)
  }

  /**
   * Calculate the CeLU function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = if (x > 0.0) x else this.alpha * (Math.exp(x / this.alpha) - 1.0)

  /**
   * Optimized derivative of the CeLU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the CeLU derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = if (fx > 0.0) 1.0 else (fx + this.alpha) / this.alpha
}
