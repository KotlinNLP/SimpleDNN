/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * Threshold(x) =
 * x, if x > threshold
 * value, otherwise
 *
 * @property threshold the threshold of the function
 * @property value the result of threshold function if x <= [threshold]
 */
class Threshold(val threshold: Double= 0.1, val value: Double = 0.0): ScalarActivationFunction() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Calculate the activation function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = if (x > this.threshold) x else this.value

  /**
   * Optimized derivative of the Threshold function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the Threshold derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double =  if (fx > this.threshold) 1.0 else this.value

}
