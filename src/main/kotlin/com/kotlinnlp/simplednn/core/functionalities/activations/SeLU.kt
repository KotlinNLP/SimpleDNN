/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * SELU(x) = scale ∗ (max(0, x) + min(0, α ∗ (exp(x) − 1)))
 *
 * References
 * [Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter, 2017, Self Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *
 * @property scale defines the scale factor. Defaults to 1.05070099
 * @property alpha defines the decreasing exponential rate for the negative values. Defaults to 1.67326324
 *
 */

class SeLU(val scale: Double = 1.05070099, val alpha: Double = 1.67326324) : ScalarActivationFunction() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check if scale and alpha are positive
   */
  init {
    require(this.scale > 1.0)
    require(this.alpha > 0.0)
  }

  /**
   * Calculate the SeLU function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = when {
    x > 0.0 -> this.scale * x
    else -> this.scale * this.alpha * (Math.exp(x) - 1.0)
  }

  /**
   * Optimized derivative of the SeLU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the SeLU derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = when {
    fx > 0.0 -> this.scale
    else -> fx + (this.scale * this.alpha)
  }
}
