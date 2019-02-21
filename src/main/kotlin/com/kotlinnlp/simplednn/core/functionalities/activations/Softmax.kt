/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The softmax function transforms the values of a vector in real numbers in the range [0.0, 1.0].
 *
 * (Note: as their sum is always 1.0 they can be interpreted as a discrete probability distribution).
 *
 * Attention!
 * This class is an optimized extension of the base Softmax.
 * It should be used as output activation of neural modules that use the cross entropy as loss function and calculate
 * the output errors as difference with the gold values.
 */
class Softmax : SoftmaxBase() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Calculate the Softmax derivative respect to the input array already activated, as optimization.
   *
   * @param fxArray the input array already activated
   *
   * @return a new NDArray containing the result
   */
  override fun dfOptimized(fxArray: DenseNDArray): DenseNDArray = DenseNDArrayFactory.ones(fxArray.shape)

  /**
   * Assign to [out] the Softmax derivative calculated respect to the input array already activated, as optimization.
   *
   * @param fxArray the input array already activated
   * @param out the NDArray in which the result is written
   */
  override fun dfOptimized(fxArray: DenseNDArray, out: DenseNDArray) {
    out.ones()
  }
}
