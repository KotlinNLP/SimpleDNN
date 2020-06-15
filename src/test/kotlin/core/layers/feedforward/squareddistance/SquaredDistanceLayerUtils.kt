/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object SquaredDistanceLayerUtils {

  /**
   *
   */
  fun buildLayer(): SquaredDistanceLayer<DenseNDArray> = SquaredDistanceLayer(
    inputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.5, -0.4))),
    outputArray = AugmentedArray(1),
    params = this.getParams35(),
    inputType =  LayerType.Input.Dense,
    dropout = 0.0
  )

  /**
   *
   */
  private fun getParams35() = SquaredDistanceLayerParameters(inputSize = 3, rank = 5).apply {
    wB.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.4, 0.6, -0.5),
        doubleArrayOf(-0.5, 0.4, 0.2),
        doubleArrayOf(0.5, 0.4, 0.1),
        doubleArrayOf(0.5, 0.2, -0.2),
        doubleArrayOf(-0.3, 0.4, 0.4)
      )))
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8))
}
