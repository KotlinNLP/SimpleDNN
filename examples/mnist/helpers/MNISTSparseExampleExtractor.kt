/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist.helpers

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonBase
import utils.SimpleExample
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.exampleextractor.ExampleExtractor
import utils.readSparseBinaryNDArrayFromDense

/**
 *
 */
class MNISTSparseExampleExtractor(val outputSize: Int) : ExampleExtractor<SimpleExample<SparseBinaryNDArray>> {

  /**
   *
   */
  override fun extract(jsonElement: JsonBase): SimpleExample<SparseBinaryNDArray> {

    val jsonArray = jsonElement as JsonArray<*>

    val features: SparseBinaryNDArray = (jsonArray[0] as JsonArray<*>).readSparseBinaryNDArrayFromDense(size = 784)
    val outputGold = DenseNDArrayFactory.oneHotEncoder(length = outputSize, oneAt = jsonArray[1] as Int)

    return SimpleExample(features, outputGold)
  }
}
