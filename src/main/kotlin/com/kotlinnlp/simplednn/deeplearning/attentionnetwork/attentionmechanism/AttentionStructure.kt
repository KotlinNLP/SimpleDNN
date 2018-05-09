/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The structure of Attention.
 *
 * @property attentionSequence the sequence of attention arrays
 * @property params the parameters of the Attention
 */
open class AttentionStructure(
  val attentionSequence: List<DenseNDArray>,
  val params: AttentionParameters
) {

  /**
   * A matrix containing the attention arrays as rows.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    values = DenseNDArrayFactory.arrayOf(this.attentionSequence.map { it.toDoubleArray() }.toTypedArray())
  )

  /**
   * The array containing the importance score.
   */
  lateinit var importanceScore: DenseNDArray

  /**
   * Check requirements.
   */
  init {

    require(this.attentionSequence.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.attentionSequence.all { it.length == this.params.attentionSize }) {
      "The attention arrays must have the expected size (%d).".format(this.params.attentionSize)
    }
  }

  /**
   * @return the errors of the attention arrays.
   */
  fun getAttentionErrors(): Array<DenseNDArray> = Array(
    size = this.attentionMatrix.values.shape.dim1,
    init = { i -> this.attentionMatrix.errors.getRow(i).t }
  )
}
