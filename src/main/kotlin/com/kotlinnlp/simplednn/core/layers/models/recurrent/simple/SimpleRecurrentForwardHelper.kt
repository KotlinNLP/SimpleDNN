/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [SimpleRecurrentLayer] in which the forward is executed
 */
class SimpleRecurrentForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   */
  override fun forward() { this.layer.params as SimpleRecurrentLayerParameters

    // y = w (dot) x + b
    this.layer.outputArray.forward(
      w = this.layer.params.unit.weights.values,
      b = this.layer.params.unit.biases.values,
      x = this.layer.inputArray.values)

    // y += wRec (dot) yPrev
    val prevStateLayer = this.layer.layerContextWindow.getPrevState()
    if (prevStateLayer != null) {
      this.layer.outputArray.addRecurrentContribution(
        parameters = this.layer.params.unit,
        prevContribution = prevStateLayer.outputArray.values)
    }

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {

    assert (this.layer.inputArray.values is DenseNDArray) {
      "Forwarding with contributions requires the input to be dense."
    }

    this.layer.params as SimpleRecurrentLayerParameters
    layerContributions as SimpleRecurrentLayerParameters

    val prevStateLayer: Layer<*>? = this.layer.layerContextWindow.getPrevState()
    val b: DenseNDArray = this.layer.params.unit.biases.values
    val bContrib: DenseNDArray = if (prevStateLayer != null) b.div(2.0) else b
    // if there's a recurrent contribution b is divided equally within the sum

    // y = w (dot) x + b ( -> b / 2)
    this.forwardArray(
      contributions = layerContributions.unit.weights.values,
      x = this.layer.inputArray.values as DenseNDArray,
      y = this.layer.outputArray.values,
      w = this.layer.params.unit.weights.values,
      b = bContrib
    )

    // y += wRec (dot) yPrev + b / 2 (recurrent contribution)
    if (prevStateLayer != null) {
      this.addRecurrentContribution(
        yPrev = prevStateLayer.outputArray.values,
        yRec = layerContributions.unit.biases.values, // a tricky way to save the recurrent contribution
        y = this.layer.outputArray.values,            //     (b.size == y.size)
        wRec = this.layer.params.unit.recurrentWeights.values,
        b = bContrib,
        contributions = layerContributions.unit.recurrentWeights.values
      )
    }

    this.layer.outputArray.activate()
  }
}
