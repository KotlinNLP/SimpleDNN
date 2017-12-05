/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Attention Layer backward helper.
 *
 * @property layer the Attention Layer Structure as support to perform calculations
 */
class AttentionLayerBackwardHelper(private val layer: AttentionLayerStructure<*>) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the errors of the output array.
   *
   *   x_i = i-th input array
   *   alpha_i = i-th value of alpha
   *   am = attention matrix
   *   gy = output errors
   *
   *   gScore_i = x_i' (dot) gy
   *   gAC = softmax_jacobian(alpha) (dot) gScore  // attention context errors
   *   gCV = am (dot) gAC  // context vector errors
   *
   *   gAM = gAC (dot) cv  // attention matrix errors
   *   gx_i = gy * alpha_i  // errors of the i-th input array
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input sequence
   */
  fun backward(paramsErrors: AttentionLayerParameters, propagateToInput: Boolean) {

    val scoreErrors: DenseNDArray = this.getScoreErrors()
    val softmaxGradients: DenseNDArray = Softmax().df(this.layer.importanceScore)
    val acErrors: DenseNDArray = softmaxGradients.dot(scoreErrors)

    paramsErrors.contextVector.values.assignValues(acErrors.t.dot(this.layer.attentionMatrix.values).t)

    if (propagateToInput) {
      this.setInputErrors()
      this.setAttentionErrors(attentionContextErrors = acErrors)
    }
  }

  /**
   * Set the errors of each array of the input sequence (which is into the structure).
   *
   *   gx_i = gy * alpha_i  // errors of the i-th input array
   */
  private fun setInputErrors() {

    val outputErrors: DenseNDArray = this.layer.outputArray.errors
    val score: DenseNDArray = this.layer.importanceScore

    for (i in 0 until this.layer.inputSequence.size) {
      this.layer.inputSequence[i].assignErrorsByProd(outputErrors, score[i])
    }
  }

  /**
   * gScore_i = x_i' (dot) gy
   *
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
   *   gAM = gAC (dot) cv
   *
   * @param attentionContextErrors the errors of the attention context.
   */
  private fun setAttentionErrors(attentionContextErrors: DenseNDArray) {
    val contextVect: DenseNDArray = this.layer.params.contextVector.values.t
    this.layer.attentionMatrix.assignErrorsByDot(attentionContextErrors, contextVect)
  }
}
