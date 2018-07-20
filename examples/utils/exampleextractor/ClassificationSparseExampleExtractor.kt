/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils.exampleextractor

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonBase
import com.kotlinnlp.simplednn.dataset.SimpleExample
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.readSparseBinaryNDArray

/**
 *
 */
class ClassificationSparseExampleExtractor(
  val inputSize: Int,
  val outputSize: Int
) : ExampleExtractor<SimpleExample<SparseBinaryNDArray>> {

  /**
   *
   */
  override fun extract(jsonElement: JsonBase): SimpleExample<SparseBinaryNDArray> {

    val jsonArray = jsonElement as JsonArray<*>

    val features: SparseBinaryNDArray = (jsonArray[0] as JsonArray<*>).readSparseBinaryNDArray(this.inputSize)
    val outputGold = DenseNDArrayFactory.oneHotEncoder(length = this.outputSize, oneAt = jsonArray[1] as Int)

    return SimpleExample(features, outputGold)
  }
}
