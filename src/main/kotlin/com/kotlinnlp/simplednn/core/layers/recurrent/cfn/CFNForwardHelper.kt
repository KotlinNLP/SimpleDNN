/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.cfn

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [CFNLayerStructure] in which the forward is executed
 */
class CFNForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: CFNLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = inG * c + f(yPrev) * forG
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

    // y += f(yPrev) * forG
    if (prevStateLayer != null) {
      val yPrev = prevStateLayer.outputArray.values

      this.layer.activatedPrevOutput = if (this.layer.activationFunction != null)
        this.layer.activationFunction.f(yPrev)
      else
        yPrev

      y.assignSum(this.layer.activatedPrevOutput!!.prod(forG))
    }
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }

  /**
   * Set gates values
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = f(wc (dot) x)
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) { this.layer.params as CFNLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val wc: DenseNDArray = this.layer.params.candidateWeights.values as DenseNDArray

    this.layer.inputGate.forward(this.layer.params.inputGate, x)
    this.layer.forgetGate.forward(this.layer.params.forgetGate, x)
    c.assignDot(wc, x)

    if (prevStateLayer != null) { // recurrent contribution for input and forget gates
      val yPrev = prevStateLayer.outputArray.valuesNotActivated
      this.layer.inputGate.addRecurrentContribution(this.layer.params.inputGate, yPrev)
      this.layer.forgetGate.addRecurrentContribution(this.layer.params.forgetGate, yPrev)
    }

    this.layer.inputGate.activate()
    this.layer.forgetGate.activate()
    this.layer.candidate.activate()
  }
}
