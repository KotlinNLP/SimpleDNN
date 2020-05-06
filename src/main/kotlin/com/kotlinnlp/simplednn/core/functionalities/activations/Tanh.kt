/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import kotlin.math.pow
import kotlin.math.tanh

/**
 * The hyperbolic tangent tanh(x) activation function is an S-shaped function,
 * transforming the values x into the range [−1, 1].
 */
object Tanh : ScalarActivationFunction {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * Calculate the activation function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = tanh(x)

  /**
   * Optimized derivative of the Tanh function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the Tanh derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = 1.0 - fx.pow(2.0)
}
