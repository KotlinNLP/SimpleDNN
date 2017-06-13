/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [RANLayerStructure] in which the forward is executed
 */
class RANForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType> {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(inG * c + yPrev * forG)
   */
  override fun forward() {

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()

    this.setGates(prevStateLayer) // must be called before accessing to the activated values of the gates

    val y: DenseNDArray = this.layer.outputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val inG: DenseNDArray = this.layer.inputGate.values
    val forG: DenseNDArray = this.layer.forgetGate.values

    // y = inG * c
    y.assignProd(inG, c)

    // y += yPrev * forG
    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.valuesNotActivated
      y.assignSum(yPrev.prod(forG))
    }

    // f(y)
    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  override fun forward(paramsContributes: LayerParameters) {
    TODO("not implemented")
  }

  /**
   * Set gates values
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = wc (dot) x + bc
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) { this.layer.params as RANLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val wc: DenseNDArray = this.layer.params.candidate.weights.values as DenseNDArray
    val bc: DenseNDArray = this.layer.params.candidate.biases.values

    this.layer.inputGate.forward(this.layer.params.inputGate, x)
    this.layer.forgetGate.forward(this.layer.params.forgetGate, x)
    c.assignDot(wc, x).assignSum(bc)

    if (prevStateLayer != null) { // recurrent contribute for input and forget gates
      val yPrev = prevStateLayer.outputArray.valuesNotActivated
      this.layer.inputGate.addRecurrentContribute(this.layer.params.inputGate, yPrev)
      this.layer.forgetGate.addRecurrentContribute(this.layer.params.forgetGate, yPrev)
    }

    this.layer.inputGate.activate()
    this.layer.forgetGate.activate()
  }
}
