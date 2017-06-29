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
 * or through the activation of a [com.kotlinnlp.simplednn.core.layers.LayerStructure]
 */
interface ActivationFunction : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Apply the activation function to [array].
   *
   * @param array the input NDArray
   *
   * @return a new NDArray containing the result
   */
  fun f(array: DenseNDArray): DenseNDArray {
    val out: DenseNDArray = array.factory.emptyArray(array.shape)
    this.f(array, out)
    return out
  }

  /**
   * Assign to [out] the result of the activation function applied to [array].
   *
   * @param array the input NDArray
   * @param out the NDArray in which the result is written
   */
  fun f(array: DenseNDArray, out: DenseNDArray)

  /**
   * Apply the activation function derivative to [xArray].
   *
   * @param xArray the input NDArray
   *
   * @return a new NDArray containing the result
   */
  fun df(xArray: DenseNDArray): DenseNDArray {
    val out: DenseNDArray = xArray.factory.emptyArray(xArray.shape)
    this.dfOptimized(this.f(xArray), out)
    return out
  }

  /**
   * Assign to [out] the activation function derivative calculated in [xArray].
   *
   * @param xArray the input NDArray
   * @param out the NDArray in which the result is written
   */
  fun df(xArray: DenseNDArray, out: DenseNDArray) {
    this.dfOptimized(this.f(xArray), out)
  }

  /**
   * Apply the activation function derivative to [fxArray].
   *
   * @param fxArray the input NDArray (WARNING: it must be f(x) for optimization)
   *
   * @return a new NDArray containing the result
   */
  fun dfOptimized(fxArray: DenseNDArray): DenseNDArray {
    val out: DenseNDArray = fxArray.factory.emptyArray(fxArray.shape)
    this.dfOptimized(fxArray, out)
    return out
  }

  /**
   * Assign to [out] the activation function derivative calculated in [fxArray].
   *
   * @param fxArray the input NDArray (WARNING: it must be f(x) for optimization)
   * @param out the NDArray in which the result is written
   */
  fun dfOptimized(fxArray: DenseNDArray, out: DenseNDArray)
}
