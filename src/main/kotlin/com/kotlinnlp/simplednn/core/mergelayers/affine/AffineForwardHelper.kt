/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.affine

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on an affine [layer].
 *
 * @property layer the [AffineLayerStructure] in which the forward is executed
 */
class AffineForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: AffineLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w1 (dot) x1 + w2 (dot) x2 + b)
   */
  override fun forward() {

    val x1: InputNDArrayType = this.layer.inputArray.values
    val x2: InputNDArrayType = this.layer.inputArray2.values

    val w1: DenseNDArray = this.layer.params.w1.values as DenseNDArray
    val w2: DenseNDArray = this.layer.params.w2.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.b.values

    this.layer.outputArray.assignValues(w1.dot(x1).assignSum(w2.dot(x2)).assignSum(b))

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * y = f(w1 (dot) x1 + w2 (dot) x2 + b)
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    layerContributions as AffineLayerParameters

    TODO("not implemented")
  }
}
