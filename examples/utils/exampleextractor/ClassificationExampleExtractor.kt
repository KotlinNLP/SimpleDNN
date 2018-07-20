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
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import utils.readDenseNDArray

/**
 *
 */
class ClassificationExampleExtractor(val outputSize: Int) : ExampleExtractor<SimpleExample<DenseNDArray>> {

  /**
   *
   */
  override fun extract(jsonElement: JsonBase): SimpleExample<DenseNDArray> {

    val jsonArray = jsonElement as JsonArray<*>

    val features: DenseNDArray = (jsonArray[0] as JsonArray<*>).readDenseNDArray()
    val outputGold = DenseNDArrayFactory.zeros(Shape(this.outputSize))

    outputGold[jsonArray[1] as Int] = 1.0

    return SimpleExample(features, outputGold)
  }
}
