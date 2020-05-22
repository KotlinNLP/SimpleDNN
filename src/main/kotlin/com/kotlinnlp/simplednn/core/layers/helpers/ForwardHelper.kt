/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @param layer the layer with which this helper works
 */
internal abstract class ForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  protected open val layer: Layer<InputNDArrayType>
) {

  /**
   * Forward the input to the output combining it with the parameters.
   */
  abstract fun forward()

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * Override this function for layers that support the relevance calculation.
   *
   * @param contributions the support in which to save the contributions of the input respect to the output
   */
  open fun forward(contributions: LayerParameters) {
    throw NotImplementedError("Forward with contributions not available for this layer.")
  }

  /**
   * Forward [x] to [y] combining it with [w] and [b], saving the contributions in [contributions].
   *
   * @param contributions a matrix which maps the contributions from each value of [x] to each value of [y]
   * @param x a [DenseNDArray]
   * @param y a [DenseNDArray]
   * @param w the weights which maps [x] to [y]
   * @param b the biases added to each value of [y] (if null no bias is added)
   */
  protected fun forwardArray(contributions: DenseNDArray,
                             x: DenseNDArray,
                             y: DenseNDArray,
                             w: DenseNDArray,
                             b: DenseNDArray? = null) {

    val xLength: Int = x.length

    for (j in 0 until w.rows) {

      y[j] = 0.0

      for (i in 0 until w.columns) {
        var contribution: Double = w[j, i] * x[i]

        if (b != null) {
          contribution += b[j] / xLength
        }

        contributions[j, i] = contribution
        y[j] += contribution
      }
    }
  }

  /**
   * Add the recurrent contribution to the output array, saving the contributions of the input in respect of the output.
   *
   * y += wRec (dot) yPrev
   *
   * @param contributions a matrix which maps the contributions from each value of [yPrev] to each value of [yRec]
   * @param yPrev the output array of the layer in the previous state
   * @param yRec the array in which the contribution coming from the recursion will be saved
   * @param y the output array of the layer
   * @param wRec the recurrent weights which connect [yPrev] to [y]
   * @param b the biases added to each value of [yRec] (if null no bias is added)
   */
  protected fun addRecurrentContribution(contributions: DenseNDArray,
                                         yPrev: DenseNDArray,
                                         yRec: DenseNDArray,
                                         y: DenseNDArray,
                                         wRec: DenseNDArray,
                                         b: DenseNDArray) {

    this.forwardArray(contributions = contributions, x = yPrev, y = yRec, w = wRec, b = b)

    y.assignSum(yRec)
  }
}
