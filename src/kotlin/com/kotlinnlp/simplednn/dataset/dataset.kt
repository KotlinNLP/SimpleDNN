/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.dataset

import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 */
interface Example

/**
 *
 * @param features features
 * @param outputGold outputGold
 */
data class SimpleExample(
  val features: NDArray,
  val outputGold: NDArray
): Example {
  /**
   *
   */
  constructor(features: DoubleArray, outputGold: DoubleArray): this(
    features = NDArray.arrayOf(features),
    outputGold = NDArray.arrayOf(outputGold)
  )
}

/**
 *
 */
data class SequenceExampleWithFinalOutput(
  val sequenceFeatures: ArrayList<NDArray>,
  val outputGold: NDArray
): Example

/**
 *
 */
data class SequenceExample(
  val sequenceFeatures: ArrayList<NDArray>,
  val sequenceOutputGold: ArrayList<NDArray>
): Example

/**
 *
 * @param features features
 * @param goldOutcomeIndex goldOutcomeIndex
 */
data class ExampleBinaryOutput(
  val features: NDArray,
  val goldOutcomeIndex: Int
): Example


/**
 *
 */
data class Corpus<Example>(
  val training: ArrayList<Example>,
  val validation: ArrayList<Example>,
  val test: ArrayList<Example>)
