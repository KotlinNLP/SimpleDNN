/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils.exampleextractor

import com.jsoniter.JsonIterator
import com.jsoniter.ValueType
import com.kotlinnlp.simplednn.dataset.SequenceExample
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import utils.readDenseNDArray

/**
 *
 */
class ClassificationSequenceExampleExtractor(val outputSize: Int) : ExampleExtractor<SequenceExample<DenseNDArray>> {

  /**
   *
   */
  override fun extract(iterator: JsonIterator): SequenceExample<DenseNDArray> {

    val featuresList = ArrayList<DenseNDArray>()
    val outputGoldList = ArrayList<DenseNDArray>()

    while (iterator.readArray()) {
      if (iterator.whatIsNext() == ValueType.ARRAY) {
        val singleExample = iterator.readDenseNDArray()
        val features = DenseNDArrayFactory.arrayOf(doubleArrayOf(singleExample[0]))
        val outputGold = DenseNDArrayFactory.zeros(Shape(11))

        outputGold[singleExample[1].toInt()] = 1.0

        featuresList.add(features)
        outputGoldList.add(outputGold)
      }
    }

    return SequenceExample(featuresList, outputGoldList)
  }
}
