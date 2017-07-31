/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.dataset.SimpleExample
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
class ClassificationExampleExtractor(val outputSize: Int) : ExampleExtractor<SimpleExample<DenseNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): SimpleExample<DenseNDArray> {

    val outputGold = DenseNDArrayFactory.zeros(Shape(this.outputSize))
    var goldIndex: Int
    var features: DenseNDArray? = null

    while (iterator.readArray()) {

      if (iterator.whatIsNext() == ValueType.ARRAY) {
        features = iterator.readDenseNDArray()

      } else if (iterator.whatIsNext() == ValueType.NUMBER) {
        goldIndex = iterator.readInt()
        outputGold[goldIndex] = 1.0
      }
    }

    return SimpleExample(features!!, outputGold)
  }
}
