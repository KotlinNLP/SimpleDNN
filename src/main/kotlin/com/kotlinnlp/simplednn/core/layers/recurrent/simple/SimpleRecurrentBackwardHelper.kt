/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which the backward is executed
 */
class SimpleRecurrentBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

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

    // TODO: implement 'meProp' algorithm

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()
    val nextStateLayer = this.layer.layerContextWindow.getNextStateLayer()

    if (nextStateLayer != null) {
      this.addLayerRecurrentGradients(nextStateLayer as SimpleRecurrentLayerStructure<*>)
    }

    this.layer.applyOutputActivationDeriv() // must be applied AFTER having added the recurrent contribution

    this.assignParamsGradients(
      paramsErrors = paramsErrors as SimpleRecurrentLayerParameters,
      prevStateOutput = prevStateLayer?.outputArray)

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * gb = gy * 1
   * gw = gb (dot) x
   * gwRec = gy (dot) yPrev
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param prevStateOutput the output array in the previous state
   */
  private fun assignParamsGradients(paramsErrors: SimpleRecurrentLayerParameters,
                                    prevStateOutput: AugmentedArray<DenseNDArray>?) {

    this.layer.outputArray.assignParamsGradients(
      paramsErrors = paramsErrors.unit,
      x = this.layer.inputArray.values,
      yPrev = prevStateOutput?.values)
  }

  /**
   * gx = gb (dot) w
   */
  private fun assignLayerGradients() { this.layer.params as SimpleRecurrentLayerParameters

    this.layer.inputArray.assignErrors(this.layer.outputArray.getInputErrors(parameters = this.layer.params.unit))
  }

  /**
   * gy += gyNext (dot) wRec
   */
  private fun addLayerRecurrentGradients(nextStateLayer: SimpleRecurrentLayerStructure<*>) {

    this.layer.params as SimpleRecurrentLayerParameters

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gRec: DenseNDArray = nextStateLayer.outputArray.getRecurrentErrors(parameters = this.layer.params.unit)

    gy.assignSum(gRec)
  }
}
