/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Attention Layer backward helper.
 *
 * @property layer the structure of the Attention Layer as support to perform calculations
 */
class AttentionLayerBackwardHelper(private val layer: AttentionLayerStructure<*>) {


  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the errors of the output array.
   *
   * @param contextVector the context vector parameter of the attention layer.
   * @param propagateToInput whether to propagate the errors to the input sequence
   */
  fun backward(contextVector: UpdatableDenseArray, propagateToInput: Boolean) {

    val scoreErrors: DenseNDArray = this.getScoreErrors()
    val softmaxGradients: DenseNDArray = Softmax().df(this.layer.importanceScore)
    val acErrors: DenseNDArray = softmaxGradients.dot(scoreErrors)
    this.layer.contextVectorErrors = acErrors.T.dot(this.layer.attentionMatrix.values).T

    if (propagateToInput) {
      this.setInputErrors()
      this.setAttentionErrors(attentionContextErrors = acErrors, contextVector = contextVector)
    }
  }

  /**
   * Set the errors of each array of the input sequence (which is into the structure).
   */
  private fun setInputErrors() {

    val outputErrors: DenseNDArray = this.layer.outputArray.errors
    val score: DenseNDArray = this.layer.importanceScore

    for (i in 0 until this.layer.inputSequence.size) {
      this.layer.inputSequence[i].errors.assignProd(outputErrors, score[i])
    }
  }

  /**
   * @return the errors of the importance score array.
   */
  private fun getScoreErrors(): DenseNDArray {

    val outputErrors: DenseNDArray = this.layer.outputArray.errors
    val scoreErrors: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(this.layer.inputSequence.size))

    for (i in 0 until this.layer.inputSequence.size) {
      val inputArray = this.layer.inputSequence[i].values
      scoreErrors[i] = inputArray.prod(outputErrors).sum()
    }

    return scoreErrors
  }

  /**
   * Set the errors of each array of the attention input sequence (which is into the structure).
   *
   * @param attentionContextErrors the errors of the attention context.
   * @param contextVector the context vector parameter of the attention layer.
   */
  private fun setAttentionErrors(attentionContextErrors: DenseNDArray, contextVector: UpdatableDenseArray) {
    this.layer.attentionMatrix.errors.assignDot(attentionContextErrors, contextVector.values.T)
  }
}
