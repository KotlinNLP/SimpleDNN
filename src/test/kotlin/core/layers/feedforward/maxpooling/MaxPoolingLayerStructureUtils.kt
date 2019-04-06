/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.maxpooling

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling.MaxPoolingLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.maxpooling.MaxPoolingLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object MaxPoolingLayerStructureUtils {

  /**
   *
   */
  private fun buildarray1(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.4, 0.1, -0.9, -0.5),
        doubleArrayOf(-0.4, 0.3, 0.7, -0.3),
        doubleArrayOf(0.8, 0.2, 0.6, 0.7),
        doubleArrayOf(0.2, -0.1, 0.6, -0.2)))

  }

  /**
   *
   */
  fun buildLayer1(): MaxPoolingLayer<DenseNDArray> {

    return MaxPoolingLayer(
        inputArray = AugmentedArray(values = buildarray1()),
        inputType = LayerType.Input.Dense,
        poolSize = Shape(2,2),
        inputSize = Shape(4,4),
        params = MaxPoolingLayerParameters())
  }
}