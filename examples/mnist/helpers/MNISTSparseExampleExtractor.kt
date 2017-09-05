/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist.helpers

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.dataset.SimpleExample
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.exampleextractor.ExampleExtractor
import utils.readSparseBinaryNDArray

/**
 *
 */
class MNISTSparseExampleExtractor(val outputSize: Int) : ExampleExtractor<SimpleExample<SparseBinaryNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): SimpleExample<SparseBinaryNDArray> {

    val outputGold = DenseNDArrayFactory.zeros(Shape(outputSize))
    var goldIndex: Int
    var features: SparseBinaryNDArray? = null

    while (iterator.readArray()) {

      if (iterator.whatIsNext() == ValueType.ARRAY) {
        features = iterator.readSparseBinaryNDArray(size = 784)

      } else if (iterator.whatIsNext() == ValueType.NUMBER) {
        goldIndex = iterator.readInt() // - 1
        outputGold[goldIndex] = 1.0
      }
    }

    return SimpleExample(features!!, outputGold)
  }
}
