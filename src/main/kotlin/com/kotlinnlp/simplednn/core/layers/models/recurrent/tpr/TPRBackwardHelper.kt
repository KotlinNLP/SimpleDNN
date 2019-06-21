/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [TPRLayer] in which the backward is executed
 */
class TPRBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: TPRLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  companion object {

    /**
     * Calculate errors of roles and symbols wrt quantization errors.
     * The quantization pushes towards attention vectors that are 1-hot.
     *
     * @property a The quantization function argument
     * @property q The weight of the quantization in the loss function
     */
    private fun getQuantizationGradients(a: DenseNDArray, q: Double): DenseNDArray {

      var s = 0.0
      val out = a.copy()

      for (i in 0 until out.length) {
        s += out[i] * out[i]
      }

      s *= 2.0

      for (i in 0 until out.length) {
        out[i] = 2.0 * out[i] * q * (2 * out[i] * out[i] - 3 * out[i] + s - 1.0)
      }

      return out
    }
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val prevStateOutput = this.layer.layerContextWindow.getPrevState()?.outputArray
    val nextStateLayer = this.layer.layerContextWindow.getNextState()

    this.addOutputRecurrentGradients(nextStateLayer as? TPRLayer<*>)

    this.assignGradients()

    this.assignParamsGradients(prevStateOutput = prevStateOutput)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun getLayerRecurrentContribution(nextStateLayer: TPRLayer<*>): DenseNDArray {

    this.layer.params as TPRLayerParameters

    val gs: DenseNDArray = nextStateLayer.aS.errors
    val gr: DenseNDArray = nextStateLayer.aR.errors

    val wRecS: DenseNDArray = this.layer.params.wRecS.values
    val wRecR: DenseNDArray = this.layer.params.wRecR.values

    val gRecS: DenseNDArray = gs.t.dot(wRecS)
    val gRecR: DenseNDArray = gr.t.dot(wRecR)

    return gRecS.assignSum(gRecR)
  }

  /**
   * Assign structure gradients
   */
  private fun assignGradients() {

    val q = this.layer.quantizationRegularizer
    this.layer.params as TPRLayerParameters

    // TODO: assign zeros only if not initialized
    this.layer.bindingMatrix.assignErrors(DenseNDArrayFactory.zeros(this.layer.bindingMatrix.values.shape))
    this.layer.bindingMatrix.errors.fromVector(this.layer.outputArray.errors)

    val gs: DenseNDArray = this.layer.r.values.t.dot(this.layer.bindingMatrix.errors.t)
    val gr: DenseNDArray = this.layer.s.values.t.dot(this.layer.bindingMatrix.errors)

    this.layer.s.assignErrors(gs)
    this.layer.r.assignErrors(gr)

    val aSactDeriv: DenseNDArray = this.layer.aS.calculateActivationDeriv()
    val aRactDeriv: DenseNDArray = this.layer.aR.calculateActivationDeriv()

    val wS: DenseNDArray = this.layer.params.S.values
    val wR: DenseNDArray = this.layer.params.R.values

    this.layer.aS
      .assignErrorsByDotT(gs, wS)
      .assignSum(getQuantizationGradients(this.layer.aS.values, q))
      .assignProd(aSactDeriv)

    this.layer.aR
      .assignErrorsByDotT(gr, wR)
      .assignSum(getQuantizationGradients(this.layer.aR.values, q))
      .assignProd(aRactDeriv)
  }

  /**
   * Assign parameters gradients
   *
   * @param prevStateOutput the outputArray in the previous state
   */
  private fun assignParamsGradients(prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.params as TPRLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val yPrev: DenseNDArray? = prevStateOutput?.values

    val gs: DenseNDArray = this.layer.s.errors
    val gr: DenseNDArray = this.layer.r.errors

    val gaS: DenseNDArray = this.layer.aS.errors
    val gaR: DenseNDArray = this.layer.aR.errors

    this.layer.params.bS.errors.values.assignValues(gaS)
    this.layer.params.bR.errors.values.assignValues(gaR)

    this.layer.params.S.errors.values.assignDot(gs, this.layer.aS.values.t)
    this.layer.params.R.errors.values.assignDot(gr, this.layer.aR.values.t)

    if (yPrev != null) {
      this.layer.params.wRecS.errors.values.assignDot(gaS, yPrev.t)
      this.layer.params.wRecR.errors.values.assignDot(gaR, yPrev.t)
    } else {
      this.layer.params.wRecS.errors.values.zeros()
      this.layer.params.wRecR.errors.values.zeros()
    }

    this.layer.params.wInS.errors.values.assignDot(gaS, x.t)
    this.layer.params.wInR.errors.values.assignDot(gaR, x.t)

  }

  /**
   * Assign layer gradients to input array
   */
  private fun assignLayerGradients() {
    this.layer.params as TPRLayerParameters

    val wInS: DenseNDArray = this.layer.params.wInS.values
    val wInR: DenseNDArray = this.layer.params.wInR.values

    val gS: DenseNDArray = this.layer.aS.errors
    val gR: DenseNDArray = this.layer.aR.errors

    this.layer.inputArray.assignErrorsByDotT(gS.t, wInS)
    this.layer.inputArray.errors.assignSum(gR.t.dot(wInR))
  }

  /**
   *
   * @param nextStateLayer the layer structure in the next state
   */
  private fun addOutputRecurrentGradients(nextStateLayer: TPRLayer<*>?) {

    if (nextStateLayer != null) {
      val gy: DenseNDArray = this.layer.outputArray.errors
      val gyRec: DenseNDArray = this.getLayerRecurrentContribution(nextStateLayer)

      gy.assignSum(gyRec)
    }
  }
}



