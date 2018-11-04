/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist.helpers

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonBase
import com.beust.klaxon.JsonObject
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import utils.SequenceExampleWithFinalOutput
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.exampleextractor.ExampleExtractor

/**
 *
 */
class MNISTSequenceExampleExtractor(val outputSize: Int)
  : ExampleExtractor<SequenceExampleWithFinalOutput<DenseNDArray>> {

  /**
   *
   */
  override fun extract(jsonElement: JsonBase): SequenceExampleWithFinalOutput<DenseNDArray> {

    val jsonObject = jsonElement as JsonObject

    val outputGold = DenseNDArrayFactory.oneHotEncoder(length = 10, oneAt = jsonObject.int("digit")!!)
    val featuresList: List<DenseNDArray> = jsonElement.array<JsonArray<*>>("sequence_data")!!.map {
      val deltaX = (it[0] as Int).toDouble()
      val deltaY = (it[1] as Int).toDouble()
      DenseNDArrayFactory.arrayOf(doubleArrayOf(deltaX, deltaY))
    }

    return SequenceExampleWithFinalOutput(featuresList, outputGold)
  }
}
