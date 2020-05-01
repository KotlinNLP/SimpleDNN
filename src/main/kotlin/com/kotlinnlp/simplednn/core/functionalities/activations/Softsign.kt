/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import java.lang.Math.abs

/**
 * The Softsign function can be considered an alternative to [Tanh],
 * transforming the values x into the range [−1, 1].
 */
class Softsign : ScalarActivationFunction() {

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
  override fun f(x: Double): Double = x / (1.0 + abs(x))

  /**
   * Optimized derivative of the Softsign function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the Softsign derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = Math.pow(1.0 - abs(fx), 2.0)
}
