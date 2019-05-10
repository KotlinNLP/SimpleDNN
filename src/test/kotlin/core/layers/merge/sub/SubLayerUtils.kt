package core.layers.merge.sub
/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object SubLayerUtils {

  /**
   *
   */
  fun buildLayer(): SubLayer<DenseNDArray> = SubLayer(
    inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.4, -0.3, 0.2))),
    inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.2, -0.6, 0.9))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 4),
    params = SubLayerParameters(inputSize = 4))

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, -0.2, 0.4, 0.0))
}