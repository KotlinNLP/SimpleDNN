/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.io.Serializable

/**
 * ActivationsFunction can either be used through an [com.kotlinnlp.simplednn.core.arrays.ActivableArray],
 * or through the activation of a [com.kotlinnlp.simplednn.core.layers.Layer]
 */
interface ActivationFunction : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Calculate the activation function respect to the input array.
   *
   * @param array the input array
   *
   * @return a new NDArray containing the result
   */
  fun f(array: DenseNDArray): DenseNDArray {

    val out: DenseNDArray = array.factory.emptyArray(array.shape)

    this.f(array, out)

    return out
  }

  /**
   * Assign to [out] the result of the activation function calculated respect to the input array.
   *
   * @param array the input array
   * @param out the NDArray in which the result is written
   */
  fun f(array: DenseNDArray, out: DenseNDArray)

  /**
   * Calculate the activation function derivative respect to the input array.
   *
   * @param xArray the input array
   *
   * @return a new NDArray containing the result
   */
  fun df(xArray: DenseNDArray): DenseNDArray {

    val out: DenseNDArray = xArray.factory.emptyArray(xArray.shape)

    this.dfOptimized(this.f(xArray), out)

    return out
  }

  /**
   * Assign to [out] the activation function derivative calculated respect to the input array.
   *
   * @param xArray the input array
   * @param out the NDArray in which the result is written
   */
  fun df(xArray: DenseNDArray, out: DenseNDArray) {

    this.dfOptimized(this.f(xArray), out)
  }

  /**
   * Calculate the activation function derivative respect to the input array already activated, as optimization.
   *
   * @param fxArray the input array already activated
   *
   * @return a new NDArray containing the result
   */
  fun dfOptimized(fxArray: DenseNDArray): DenseNDArray {

    val out: DenseNDArray = fxArray.factory.emptyArray(fxArray.shape)

    this.dfOptimized(fxArray, out)

    return out
  }

  /**
   * Assign to [out] the activation function derivative calculated respect to the input array already activated, as
   * optimization.
   *
   * @param fxArray the input array already activated
   * @param out the NDArray in which the result is written
   */
  fun dfOptimized(fxArray: DenseNDArray, out: DenseNDArray)
}
