/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [RANLayerStructure] in which the forward is executed
 */
class RANForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output through the gates, combining it with the parameters.
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
   * Forward the input to the output through the gates, combining it with the parameters and saving the contributions
   * of the input array in respect of each gate.
   *
   * y = f(inG * c + yPrev * forG)
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()

    // must be called before accessing to the activated values of the gates
    this.setGates(prevStateLayer = prevStateLayer, layerContributions = layerContributions as RANLayerParameters)

    val y: DenseNDArray = this.layer.outputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val inG: DenseNDArray = this.layer.inputGate.values
    val forG: DenseNDArray = this.layer.forgetGate.values

    // y = inG * c
    y.assignProd(inG, c)

    // y += yPrev * forG
    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.valuesNotActivated
      val yRec: DenseNDArray = layerContributions.candidate.biases.values as DenseNDArray
      // a tricky way to save the contributions coming from recursion (b.size == y.size)

      yRec.assignProd(yPrev, forG) // save contribution coming from recursion
      y.assignSum(yRec)
    }

    // f(y)
    this.layer.outputArray.activate()
  }

  /**
   * Set gates values.
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = wc (dot) x + bc
   *
   * @param prevStateLayer the layer in the previous state
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) { this.layer.params as RANLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values

    this.layer.inputGate.forward(parameters = this.layer.params.inputGate, x = x)
    this.layer.forgetGate.forward(parameters = this.layer.params.forgetGate, x = x)
    this.layer.candidate.forward(parameters = this.layer.params.candidate, x = x)

    if (prevStateLayer != null) { // recurrent contribution for input and forget gates
      val yPrev = prevStateLayer.outputArray.valuesNotActivated
      this.layer.inputGate.addRecurrentContribution(this.layer.params.inputGate, yPrev)
      this.layer.forgetGate.addRecurrentContribution(this.layer.params.forgetGate, yPrev)
    }

    this.layer.inputGate.activate()
    this.layer.forgetGate.activate()
  }

  /**
   * Set gates values, saving the contributions of the input in respect of the output.
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = wc (dot) x + bc
   *
   * @param prevStateLayer the layer in the previous state
   * @param layerContributions the structure in which to save the contributions of the input in respect of the output
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?, layerContributions: RANLayerParameters) {
    this.layer.params as RANLayerParameters

    // biases are divided equally within the sum if there's a contribution coming from recursion
    val splitBiases: Boolean = prevStateLayer != null
    val inGParams: RecurrentParametersUnit = this.layer.params.inputGate
    val forGParams: RecurrentParametersUnit = this.layer.params.forgetGate
    val bInGBase: DenseNDArray = inGParams.biases.values as DenseNDArray
    val bForGBase: DenseNDArray = forGParams.biases.values as DenseNDArray
    val bInG: DenseNDArray = if (splitBiases) bInGBase.div(2.0) else bInGBase
    val bForG: DenseNDArray = if (splitBiases) bForGBase.div(2.0) else bForGBase

    this.forwardGates(layerContributions = layerContributions, bInG = bInG, bForG = bForG)

    if (prevStateLayer != null) { // recurrent contribution for input and forget gates
      this.addGatesRecurrentContribution(
        layerContributions = layerContributions,
        yPrev = prevStateLayer.outputArray.valuesNotActivated,
        bInG = bInG,
        bForG = bForG)
    }

    this.layer.inputGate.activate()
    this.layer.forgetGate.activate()
  }

  /**
   * Forward the input to the gates, saving its contributions in respect of each gate.
   *
   * g += wRec (dot) yPrev
   *
   * @param layerContributions the structure in which to save the contributions of the input in respect of each gate
   * @param bInG the biases array of the input gate
   * @param bForG the biases array of the forget gate
   */
  private fun forwardGates(layerContributions: RANLayerParameters, bInG: DenseNDArray, bForG: DenseNDArray) {
    this.layer.params as RANLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val candidateParams: ParametersUnit = this.layer.params.candidate

    this.forwardArray(
      contributions = layerContributions.candidate.weights.values,
      x = x,
      y = this.layer.candidate.values,
      w = candidateParams.weights.values as DenseNDArray,
      b = candidateParams.biases.values as DenseNDArray)

    this.forwardArray(
      contributions = layerContributions.inputGate.weights.values,
      x = x,
      y = this.layer.inputGate.values,
      w = this.layer.params.inputGate.weights.values as DenseNDArray,
      b = bInG)

    this.forwardArray(
      contributions = layerContributions.forgetGate.weights.values,
      x = x,
      y = this.layer.forgetGate.values,
      w = this.layer.params.forgetGate.weights.values as DenseNDArray,
      b = bForG)
  }

  /**
   * Add the recurrent contribution to the gate, saving the contributions of the input in respect of the output.
   *
   * g += wRec (dot) yPrev
   *
   * @param yPrev the output array of the layer in the previous state
   * @param layerContributions the structure in which to save the contributions of the input in respect of each gate
   * @param bInG the biases array of the input gate
   * @param bForG the biases array of the forget gate
   */
  private fun addGatesRecurrentContribution(yPrev: DenseNDArray,
                                            bInG: DenseNDArray,
                                            bForG: DenseNDArray,
                                            layerContributions: RANLayerParameters) {

    this.layer.params as RANLayerParameters

    val inGParams: RecurrentParametersUnit = this.layer.params.inputGate
    val forGParams: RecurrentParametersUnit = this.layer.params.forgetGate

    this.addRecurrentContribution(
      yPrev = yPrev,
      yRec = layerContributions.inputGate.biases.values as DenseNDArray, // a tricky way to save the contribution coming
      y = this.layer.inputGate.values,                                   // from recursion (b.size == y.size)
      wRec = inGParams.recurrentWeights.values,
      b = bInG,
      contributions = layerContributions.inputGate.recurrentWeights.values
    )

    this.addRecurrentContribution(
      yPrev = yPrev,
      yRec = layerContributions.forgetGate.biases.values as DenseNDArray, // a tricky way to save the contribution
      y = this.layer.forgetGate.values,                                   // coming from recursion (b.size == y.size)
      wRec = forGParams.recurrentWeights.values,
      b = bForG,
      contributions = layerContributions.forgetGate.recurrentWeights.values
    )
  }
}
