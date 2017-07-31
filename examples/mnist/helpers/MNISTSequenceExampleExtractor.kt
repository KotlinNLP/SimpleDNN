/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist.helpers

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import ExampleExtractor
import com.kotlinnlp.simplednn.dataset.SequenceExampleWithFinalOutput
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import readDenseNDArray

/**
 *
 */
class MNISTSequenceExampleExtractor(val outputSize: Int)
  : ExampleExtractor<SequenceExampleWithFinalOutput<DenseNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): SequenceExampleWithFinalOutput<DenseNDArray> {

    val featuresList = ArrayList<DenseNDArray>()
    val outputGold = DenseNDArrayFactory.zeros(Shape(10))

    // read "digit"
    iterator.readObject()
    outputGold[iterator.readInt()] = 1.0

    // skip "id"
    iterator.readObject()
    iterator.readAny()

    // read "sequence_data"
    iterator.readObject()

    while (iterator.readArray()) {
      if (iterator.whatIsNext() == ValueType.ARRAY) {
        val features = iterator.readDenseNDArray()
        val deltaX = features[0]
        val deltaY = features[1]
        featuresList.add(DenseNDArrayFactory.arrayOf(doubleArrayOf(deltaX, deltaY)))
      }
    }

    return SequenceExampleWithFinalOutput(featuresList, outputGold)
  }
}
