/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.distance

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import java.lang.Math.abs
import kotlin.math.exp

/**
 * The helper which executes the forward on a [DistanceLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class DistanceForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: DistanceLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer){

  /**
   * Forward the input to the output calculating a score value d ∈ [0, 1]. d = exp(-||input1-input2||1)
   */
  override fun forward() {

    val diffVector : DenseNDArray = DenseNDArrayFactory.fromNDArray(this.layer.inputArray1.values)
    var sum = 0.0
    val outputScore = DoubleArray(1)

    diffVector.assignSub(this.layer.inputArray2.values)

    diffVector.toDoubleArray().forEach { element -> sum += abs(element) }
    outputScore[0] = exp(-sum)
    this.layer.outputArray.assignValues(DenseNDArrayFactory.arrayOf(outputScore))

  }

  /**
   * Forward the input to the output saving the contributions.
   * Not available for the Sum layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Sum layer.")
  }
}