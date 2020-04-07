/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concat

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [ConcatLayer].
 *
 * @property layer the layer in which the backward is executed
 */
internal class ConcatBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ConcatLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * The errors split sizes.
   */
  private val errorsSplitSizes = this.layer.params.inputsSize.toIntArray()

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign the the layer gradients.
   */
  private fun assignLayerGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.inputArrays.zip(gy.splitV(*this.errorsSplitSizes)).forEach { (x, gradients) ->
      x.assignErrors(gradients)
    }
  }
}
