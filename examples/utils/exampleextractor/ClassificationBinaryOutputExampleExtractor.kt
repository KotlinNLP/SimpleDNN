/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils.exampleextractor

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.dataset.BinaryOutputExample
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.readDenseNDArray

/**
 *
 */
class ClassificationBinaryOutputExampleExtractor : ExampleExtractor<BinaryOutputExample<DenseNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): BinaryOutputExample<DenseNDArray> {

    var goldIndex = 0
    var features: DenseNDArray? = null

    while (iterator.readArray()) {

      if (iterator.whatIsNext() == ValueType.ARRAY) {
        features = iterator.readDenseNDArray()

      } else if (iterator.whatIsNext() == ValueType.NUMBER) {
        goldIndex = iterator.readInt()
      }
    }

    return BinaryOutputExample(features!!, goldIndex)
  }
}
