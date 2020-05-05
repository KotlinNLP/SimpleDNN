/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.lang.RuntimeException
import kotlin.math.*

/**
 * GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
 *
 * It is a smoother version of the [ReLU].
 *
 * Reference:
 * [Dan Hendrycks, Kevin Gimpel, 2018, Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
 */
object GeLU : ScalarActivationFunction() {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * Calculate the GeLU function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Double): Double = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x.pow(3.0))))

  /**
   * Derivative of the GeLU function, calculated in [x].
   *
   * @param x the input
   *
   * @return the GeLU derivative calculated in x
   */
  override fun df(x: Double): Double {

    fun sech(x: Double): Double = 2.0 * exp(-x) / (1.0 + exp(-2.0 * x))

    val x3 = x.pow(3.0)
    val a = 0.0356774 * x3 + 0.797885 * x
    val s = sech(a)

    return 0.5 * tanh(a) + (0.0535161 * x3 + 0.398942 * x) * s * s + 0.5
  }

  /**
   * Calculate the activation function derivative respect to the input array.
   *
   * @param xArray the input array
   *
   * @return a new NDArray containing the result
   */
  override fun df(xArray: DenseNDArray): DenseNDArray {

    val out: DenseNDArray = xArray.factory.emptyArray(xArray.shape)

    this.df(xArray, out)

    return out
  }

  /**
   * Optimized derivative of the GeLU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the GeLU derivative calculated in x
   */
  override fun dfOptimized(fx: Double): Double = throw RuntimeException("Optimized derivative not available for GeLU")
}
