/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.exp

/**
 * The softmax function transforms the values of a vector in real numbers in the range [0.0, 1.0].
 *
 * (Note: as their sum is always 1.0 they can be interpreted as a discrete probability distribution).
 */
open class SoftmaxBase : ActivationFunction {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Assign to [out] the result of Softmax function applied to [array].
   *
   * @param array the input NDArray
   * @param out the NDArray in which the result is written
   */
  override fun f(array: DenseNDArray, out: DenseNDArray) {

    val max: Double = array.max()
    var sum = 0.0

    for (i in 0 until array.length) {
      val e: Double = exp(array[i] - max)
      out[i] = e
      sum += e
    }

    out.assignDiv(sum)
  }

  /**
   * Calculate the Softmax derivative respect to the input array already activated, as optimization.
   *
   * @param fxArray the input array already activated
   *
   * @return a new NDArray containing the result
   */
  override fun dfOptimized(fxArray: DenseNDArray): DenseNDArray {

    val jacobianMatrix: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(fxArray.length, fxArray.length))

    this.dfOptimized(fxArray = fxArray, out = jacobianMatrix)

    return jacobianMatrix
  }

  /**
   * Assign to [out] the Softmax derivative calculated respect to the input array already activated, as optimization.
   *
   * @param fxArray the input array already activated
   * @param out the NDArray in which the result is written
   */
  override fun dfOptimized(fxArray: DenseNDArray, out: DenseNDArray) {

    for (i in 0 until fxArray.length) {
      for (j in i until fxArray.length) {

        out[i, j] = if (i == j) fxArray[i] * (1.0 - fxArray[j]) else - fxArray[i] * fxArray[j]

        if (i != j) {
          out[j, i] = out[i, j]
        }
      }
    }
  }
}
