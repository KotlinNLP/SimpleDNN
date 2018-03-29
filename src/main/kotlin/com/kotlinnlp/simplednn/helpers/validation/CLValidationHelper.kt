/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.dataset.BinaryOutputExample
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.CLNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A helper which executes the validation of a [CLNetwork] with a dataset of [BinaryOutputExample]s.
 *
 * @param network the competitive learning network
 */
class CLValidationHelper(private val network: CLNetwork) : ValidationHelper<BinaryOutputExample<DenseNDArray>>() {

  /**
   * Validate a single example.
   *
   * @param example the example to validate
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   *
   * @return a Boolean indicating if the predicted output matches the gold output
   */
  override fun validate(example: BinaryOutputExample<DenseNDArray>, saveContributions: Boolean): Boolean {

    val output = this.network.classify(inputArray = example.features)

    return output == example.goldOutcomeIndex
  }
}
