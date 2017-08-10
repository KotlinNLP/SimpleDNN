/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * A support structure for the DeltaRNN, used to save temporary results during a forward and using them to calculate the
 * relevance later.
 *
 * @property outputSize the size of the output array of the layer
 */
data class DeltaRNNRelevanceSupport(val outputSize: Int) {

  /**
   * The contribution from the input to the d1 array, including half biases of the candidate.
   */
  val d1Input = AugmentedArray<DenseNDArray>(values = DenseNDArrayFactory.emptyArray(Shape(this.outputSize)))

  /**
   * The contribution from the previous state to the d1 array, including half biases of the candidate.
   */
  val d1Rec = AugmentedArray<DenseNDArray>(values = DenseNDArrayFactory.emptyArray(Shape(this.outputSize)))

  /**
   * The d2 array.
   */
  val d2 = AugmentedArray<DenseNDArray>(values = DenseNDArrayFactory.emptyArray(Shape(this.outputSize)))
}
