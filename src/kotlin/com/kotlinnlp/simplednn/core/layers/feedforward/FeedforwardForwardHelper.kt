/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [FeedforwardLayerStructure] in which the forward is executed
 */
class FeedforwardForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b)
   */
  override fun forward() { this.layer.params as FeedforwardLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values

    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.biases.values

    y.assignDot(w, x).assignSum(b)

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * y = f(w (dot) x + b)
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  override fun forward(paramsContributes: LayerParameters) {
    this.layer.params as FeedforwardLayerParameters
    paramsContributes as FeedforwardLayerParameters

    this.forwardArray(
      x = this.layer.inputArray.values,
      y = this.layer.outputArray.values,
      w = this.layer.params.weights.values as DenseNDArray,
      b = this.layer.params.biases.values,
      contributes = paramsContributes.weights.values
    )

    this.layer.outputArray.activate()
  }
}