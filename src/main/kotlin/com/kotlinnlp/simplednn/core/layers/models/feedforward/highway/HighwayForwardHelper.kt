/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.highway

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [HighwayLayer] in which the forward is executed
 */
class HighwayForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: HighwayLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * T = sigmoid(wT (dot) x + bT)
   * y = T * f(wIn (dot) x + bIn) + (1 - T) * x
   */
  override fun forward() { this.layer.params as HighwayLayerParameters

    // TODO: extend for all input types
    require(this.layer.inputArray.values is DenseNDArray) { "Highway layer supports only dense input." }

    val x: InputNDArrayType = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values
    val inputUnit: DenseNDArray = this.layer.inputUnit.values
    val tGate: DenseNDArray = this.layer.transformGate.values

    this.layer.inputUnit.forward(
      w = this.layer.params.input.weights.values,
      b = this.layer.params.input.biases.values,
      x = x
    )

    this.layer.inputUnit.activate()

    this.layer.transformGate.forward(
      w = this.layer.params.transformGate.weights.values,
      b = this.layer.params.transformGate.biases.values,
      x = x
    )

    this.layer.transformGate.activate()

    y.assignProd(tGate, inputUnit).assignSum(tGate.reverseSub(1.0).assignProd(x as DenseNDArray))
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }
}
