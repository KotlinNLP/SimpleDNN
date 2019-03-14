/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.sqrt

/**
 * The helper which executes the forward on a [CosineLayer].
 *
 * @property layer the layer in which the forward is executed
 */
class CosineForwardHelper (override val layer: CosineLayer) : ForwardHelper<DenseNDArray>(layer) {


  /**
   * Forward the input to the output calculating a score value d ∈ [-1, 1]. d = cosine_similarity(input1-input2)
   * cosine_similarity(input1, input2) = (input1 dot input2) / (||input1||2 * ||input2||2)
   */
  override fun forward() {

    val dotProduct = this.layer.inputArray1.values.t.dot(this.layer.inputArray2.values)[0]

    var input1Norm = 0.0
    var input2Norm = 0.0

    val outputScore = DoubleArray(1)

    (0 until this.layer.inputArray1.values.length).forEach { i ->
      input1Norm += this.layer.inputArray1.values[i] * this.layer.inputArray1.values[i]
      input2Norm += this.layer.inputArray2.values[i] * this.layer.inputArray2.values[i]
    }

    input1Norm = sqrt(input1Norm)
    input2Norm = sqrt(input2Norm)

    outputScore[0] = dotProduct / (input1Norm * input2Norm)
    this.layer.input1Norm = input1Norm
    this.layer.input2Norm = input2Norm
    this.layer.outputArray.assignValues(DenseNDArrayFactory.arrayOf(outputScore))

  }

  /**
   * Forward the input to the output saving the contributions.
   * Not available for the Cosine layer.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Distance layer.")
  }

}