/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [RANLayerStructure] in which the backward is executed
 */
class RANBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: RANLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * The masks used to execute the 'meProp' propagation algorithm.
   *
   * @param input the mask of the k nodes of the input gate with the top errors
   * @param forget the mask of the k nodes of the forget gate with the top errors
   * @param candidate the mask of the k nodes of the candidate with the top errors
   */
  private data class MePropMasks(val input: NDArrayMask, val forget: NDArrayMask, val candidate: NDArrayMask)

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean, mePropK: Double?) {

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer() as? RANLayerStructure
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer() as? RANLayerStructure

    this.layer.applyOutputActivationDeriv() // must be applied BEFORE having added the recurrent contribution

    this.addOutputRecurrentGradients(nextStateLayer = nextStateLayer, mePropK = mePropK)

    this.assignGatesGradients(prevStateLayer)

    // Be careful: the masks must be get after assigning gates gradients.
    val mePropMasks: MePropMasks? = if (mePropK != null) this.getGateMasks(mePropK) else null

    this.assignParamsGradients(
      paramsErrors = paramsErrors as RANLayerParameters,
      prevStateOutput = prevStateLayer?.outputArray,
      mePropMasks = mePropMasks)

    if (propagateToInput) {
      this.assignLayerGradients(mePropMasks)
    }
  }

  /**
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors
   *
   * @return the gate masks used to execute the 'meProp' propagation algorithm
   */
  private fun getGateMasks(mePropK: Double) = MePropMasks(
    input = this.layer.inputGate.getMePropMask(mePropK),
    forget = this.layer.forgetGate.getMePropMask(mePropK),
    candidate = this.layer.candidate.getMePropMask(mePropK))

  /**
   * @param prevStateLayer the layer in the previous state
   */
  private fun assignGatesGradients(prevStateLayer: RANLayerStructure<*>?) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    val inputGate = this.layer.inputGate
    val forgetGate = this.layer.forgetGate
    val candidate = this.layer.candidate

    val inG: DenseNDArray = inputGate.values
    val c: DenseNDArray = candidate.values

    val inGDeriv: DenseNDArray = inputGate.calculateActivationDeriv()

    this.layer.inputGate.assignErrorsByProd(c, inGDeriv).assignProd(gy)
    this.layer.candidate.assignErrorsByProd(inG, gy)

    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values
      val forGDeriv: DenseNDArray = forgetGate.calculateActivationDeriv()

      this.layer.forgetGate.assignErrorsByProd(yPrev, forGDeriv).assignProd(gy)

    } else {
      this.layer.forgetGate.assignZeroErrors()
    }
  }

  /**
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the outputArray in the previous state
   * @param mePropMasks the gate masks used to execute the 'meProp' propagation algorithm
   */
  private fun assignParamsGradients(paramsErrors: RANLayerParameters,
                                    prevStateOutput: AugmentedArray<DenseNDArray>?,
                                    mePropMasks: MePropMasks?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.valuesNotActivated

    this.layer.inputGate.assignParamsGradients(
      paramsErrors = paramsErrors.inputGate, x = x, yPrev = yPrev, mePropMask = mePropMasks?.input)

    this.layer.forgetGate.assignParamsGradients(
      paramsErrors = paramsErrors.forgetGate, x = x, yPrev = yPrev, mePropMask = mePropMasks?.forget)

    this.layer.candidate.assignParamsGradients(
      paramsErrors = paramsErrors.candidate, x = x, mePropMask = mePropMasks?.candidate)
  }

  /**
   * @param mePropMasks the gate masks used to execute the 'meProp' propagation algorithm
   */
  private fun assignLayerGradients(mePropMasks: MePropMasks?) {

    this.layer.params as RANLayerParameters

    this.layer.inputArray
      .assignErrors(this.layer.inputGate.getInputErrors(
        parameters = this.layer.params.inputGate, mePropMask = mePropMasks?.input))
      .assignSum(this.layer.forgetGate.getInputErrors(
        parameters = this.layer.params.forgetGate, mePropMask = mePropMasks?.forget))
      .assignSum(this.layer.candidate.getInputErrors(
        parameters = this.layer.params.candidate, mePropMask = mePropMasks?.candidate))
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  private fun addOutputRecurrentGradients(nextStateLayer: RANLayerStructure<*>?, mePropK: Double?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer = nextStateLayer, mePropK = mePropK)

      gy.assignSum(gyRec)
    }
  }

  /**
   * @param nextStateLayer the layer structure in the next state
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  private fun getLayerRecurrentContribution(nextStateLayer: RANLayerStructure<*>, mePropK: Double?): DenseNDArray {

    this.layer.params as RANLayerParameters

    val gyNext: DenseNDArray = nextStateLayer.outputArray.errors
    val forG: DenseNDArray = nextStateLayer.forgetGate.values

    val gRec1: DenseNDArray = forG.assignProd(gyNext)

    val gRec2: DenseNDArray = nextStateLayer.inputGate.getRecurrentErrors(
      parameters = this.layer.params.inputGate,
      mePropMask = if (mePropK != null) nextStateLayer.inputGate.getMePropMask(mePropK) else null)

    val gRec3: DenseNDArray = nextStateLayer.forgetGate.getRecurrentErrors(
      parameters = this.layer.params.forgetGate,
      mePropMask = if (mePropK != null) nextStateLayer.forgetGate.getMePropMask(mePropK) else null)

    return gRec1.assignSum(gRec2).assignSum(gRec3)
  }
}
