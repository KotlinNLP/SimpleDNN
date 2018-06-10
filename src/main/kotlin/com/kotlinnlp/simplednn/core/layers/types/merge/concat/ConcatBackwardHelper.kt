/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.merge.concat

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.utils.SplitVHelper

/**
 * The helper which executes the backward on a [ConcatLayerStructure].
 *
 * @property layer the layer in which the backward is executed
 */
class ConcatBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ConcatLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * The backward helper to split the output errors.
   */
  private val errorsSplitter = SplitVHelper(*this.layer.params.inputsSize.toIntArray())

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

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign the the layer gradients.
   */
  private fun assignLayerGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.inputArrays.zip(this.errorsSplitter.split(gy)).forEach { (x, gradients) ->
      x.assignErrors(gradients)
    }
  }
}
