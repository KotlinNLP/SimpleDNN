/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.sub

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [SubLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class SubForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SubLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output subtracting 2nd input array from 1st input arrays.
   */
  override fun forward() {

    val sub: DenseNDArray = this.layer.inputArray1.values.sub(this.layer.inputArray2.values) as DenseNDArray
    this.layer.outputArray.assignValues(sub)
  }

  /**
   * Forward the input to the output saving the contributions.
   * Not available for the Sub layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Sum layer.")
  }
}
