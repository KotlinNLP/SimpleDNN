/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm

import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [LTMLayer] in which the backward is executed
 */
class LTMBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: LTMLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevState() as? LTMLayer
    val nextStateLayer = this.layer.layerContextWindow.getNextState() as? LTMLayer

    if (nextStateLayer != null) this.addOutputRecurrentGradients(nextStateLayer)

    this.assignCellGradients(nextStateLayer)
    this.assignGatesGradients()

    this.assignParamsGradients()

    // Note: the previous layer will use the input gradients of this layer because they are equal to the recurrent
    // error of the output.
    if (propagateToInput || prevStateLayer != null) {
      this.assignInputGradients()
    }
  }

  /**
   * @param nextStateLayer the layer in the next state
   */
  private fun assignCellGradients(nextStateLayer: LTMLayer<*>?) {

    val l3: DenseNDArray = this.layer.inputGate3.values
    val gy: DenseNDArray = this.layer.outputArray.errors
    val cellDeriv: DenseNDArray = this.layer.cell.calculateActivationDeriv()

    this.layer.cell.assignErrorsByProd(gy, l3)

    if (nextStateLayer != null)
      this.layer.cell.errors.assignSum(nextStateLayer.c.errors)

    this.layer.cell.errors.assignProd(cellDeriv)

    val wCell: DenseNDArray = this.layer.params.cell.weights.values
    this.layer.c.assignErrors(this.layer.cell.getInputErrors(wCell))
  }

  /**
   * Assign the gradients of the gates.
   */
  private fun assignGatesGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gC: DenseNDArray = this.layer.c.errors

    val l1: DenseNDArray = this.layer.inputGate1.values
    val l2: DenseNDArray = this.layer.inputGate2.values
    val cell: DenseNDArray = this.layer.cell.values

    val l1Deriv: DenseNDArray = this.layer.inputGate1.calculateActivationDeriv()
    val l2Deriv: DenseNDArray = this.layer.inputGate2.calculateActivationDeriv()
    val l3Deriv: DenseNDArray = this.layer.inputGate3.calculateActivationDeriv()

    this.layer.inputGate1.assignErrorsByProd(gC, l1Deriv.assignProd(l2))
    this.layer.inputGate2.assignErrorsByProd(gC, l2Deriv.assignProd(l1))
    this.layer.inputGate3.assignErrorsByProd(gy, l3Deriv.assignProd(cell))
  }

  /**
   * Assign the gradients of the parameters.
   */
  private fun assignParamsGradients() {

    this.layer.inputGate1.assignParamsGradients(
      gw = this.layer.params.inputGate1.weights.errors.values,
      gb = this.layer.params.inputGate1.biases.errors.values,
      x = this.layer.x)

    this.layer.inputGate2.assignParamsGradients(
      gw = this.layer.params.inputGate2.weights.errors.values,
      gb = this.layer.params.inputGate2.biases.errors.values,
      x = this.layer.x)

    this.layer.inputGate3.assignParamsGradients(
      gw = this.layer.params.inputGate3.weights.errors.values,
      gb = this.layer.params.inputGate3.biases.errors.values,
      x = this.layer.x)

    this.layer.cell.assignParamsGradients(
      gw = this.layer.params.cell.weights.errors.values,
      gb = this.layer.params.cell.biases.errors.values,
      x = this.layer.c.values)
  }

  /**
   * Add output gradients coming from the next state.
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: LTMLayer<*>) {

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyRec: DenseNDArray = nextStateLayer.inputArray.errors

    gy.assignSum(gyRec)
  }

  /**
   * Assign the gradients of the input.
   */
  private fun assignInputGradients() {

    val w1: DenseNDArray = this.layer.params.inputGate1.weights.values
    val w2: DenseNDArray = this.layer.params.inputGate2.weights.values
    val w3: DenseNDArray = this.layer.params.inputGate3.weights.values

    val gL1: DenseNDArray = this.layer.inputGate1.getInputErrors(w1)
    val gL2: DenseNDArray = this.layer.inputGate2.getInputErrors(w2)
    val gL3: DenseNDArray = this.layer.inputGate3.getInputErrors(w3)

    this.layer.inputArray.assignErrors(gL1.assignSum(gL2).assignSum(gL3))
  }
}
