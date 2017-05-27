/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.dataset

import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 *
 */
interface Example

/**
 *
 * @param features features
 * @param outputGold outputGold
 */
data class SimpleExample<FeaturesNDArrayType: NDArray<FeaturesNDArrayType>>(
  val features: FeaturesNDArrayType,
  val outputGold: DenseNDArray
): Example {
  /**
   *
   */
  companion object {
    operator fun invoke(features: DoubleArray, outputGold: DoubleArray) = SimpleExample(
      features = DenseNDArrayFactory.arrayOf(features),
      outputGold = DenseNDArrayFactory.arrayOf(outputGold)
    )
  }
}

/**
 *
 */
data class SequenceExampleWithFinalOutput<FeaturesNDArrayType: NDArray<FeaturesNDArrayType>>(
  val sequenceFeatures: ArrayList<FeaturesNDArrayType>,
  val outputGold: DenseNDArray
): Example

/**
 *
 */
data class SequenceExample<FeaturesNDArrayType: NDArray<FeaturesNDArrayType>>(
  val sequenceFeatures: ArrayList<FeaturesNDArrayType>,
  val sequenceOutputGold: ArrayList<DenseNDArray>
): Example

/**
 *
 * @param features features
 * @param goldOutcomeIndex goldOutcomeIndex
 */
data class ExampleBinaryOutputSequenceExample<FeaturesNDArrayType: NDArray<FeaturesNDArrayType>>(
  val features: FeaturesNDArrayType,
  val goldOutcomeIndex: Int
): Example


/**
 *
 */
data class Corpus<Example>(
  val training: ArrayList<Example>,
  val validation: ArrayList<Example>,
  val test: ArrayList<Example>)
