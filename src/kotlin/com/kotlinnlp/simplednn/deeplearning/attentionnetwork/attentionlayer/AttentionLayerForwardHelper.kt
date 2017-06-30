/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Attention Layer forward helper.
 *
 * @property layer the structure of the Attention Layer as support to perform calculations
 */
class AttentionLayerForwardHelper(private val layer: AttentionLayerStructure<*>) {

  /**
   * Perform the forward of the input sequence contained into the [layer].
   *
   * @param contextVector the context vector parameter of the attention layer.
   *
   * @return the output array
   */
  fun forward(contextVector: UpdatableDenseArray) {

    this.layer.attentionContext = this.layer.attentionMatrix.values.dot(contextVector.values)

    this.layer.importanceScore = Softmax().f(this.layer.attentionContext)

    this.calculateOutput()
  }

  /**
   * Calculate the values of the output array.
   */
  private fun calculateOutput() {

    val y: DenseNDArray = this.layer.outputArray.values

    y.zeros()

    this.layer.inputSequence.forEachIndexed { i, inputArray ->
      y.assignSum(inputArray.values.prod(this.layer.importanceScore[i]))
    }
  }
}
