/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [AttentionLayer] in which the forward is executed
 */
internal class AttentionForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: AttentionLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * Calculate the values of the output array.
   *
   *   y = sum by { x_i * alpha_i }
   */
  override fun forward() { this.layer.params as AttentionMechanismLayerParameters

    this.layer.attentionMechanism.forward()

    val y: DenseNDArray = this.layer.outputArray.values

    y.zeros()

    this.layer.inputArrays.forEachIndexed { i, inputArray ->
      y.assignSum(inputArray.values.prod(this.layer.attentionMechanism.outputArray.values[i]))
    }
  }
}
