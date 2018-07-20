/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils.exampleextractor

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonBase
import com.kotlinnlp.simplednn.dataset.SequenceExample
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
class ClassificationSequenceExampleExtractor(val outputSize: Int) : ExampleExtractor<SequenceExample<DenseNDArray>> {

  /**
   *
   */
  override fun extract(jsonElement: JsonBase): SequenceExample<DenseNDArray> {

    val jsonArray = jsonElement as JsonArray<*>

    val featuresList = ArrayList<DenseNDArray>()
    val outputGoldList = ArrayList<DenseNDArray>()

    jsonArray.forEach {
      it as JsonArray<*>

      featuresList.add(DenseNDArrayFactory.arrayOf(doubleArrayOf(it[0] as Double)))
      outputGoldList.add(DenseNDArrayFactory.oneHotEncoder(length = 11, oneAt = it[1] as Int))
    }

    return SequenceExample(featuresList, outputGoldList)
  }
}
