/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper that implements the Attention Mechanism used by the Attention Layer.
 * It provides the forward and the backward methods.
 */
object AttentionMechanism {

  /**
   * Perform the forward of the Attention Mechanism.
   *
   *   am = attention matrix
   *   cv = context vector
   *
   *   ac = am (dot) cv  // attention context
   *   alpha = softmax(ac)  // importance score
   *
   * @return the importance score
   */
  fun forward(attentionMatrix: DenseNDArray, contextVector: DenseNDArray): DenseNDArray {

    val attentionContext: DenseNDArray = attentionMatrix.dot(contextVector)

    return Softmax().f(attentionContext)
  }

  /**
   * Executes the backward assigning the errors of the context vector and the attention matrix.
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
   * @param importanceScore the importance score
   * @param importanceScoreErrors the errors of the importance score
   * @param attentionMatrix the attention matrix
   * @param contextVector the context vector
   * @param contextVectorErrors the errors of the context vector
   */
  fun backward(
    importanceScore: DenseNDArray,
    importanceScoreErrors: DenseNDArray,
    attentionMatrix: AugmentedArray<DenseNDArray>,
    contextVector: DenseNDArray,
    contextVectorErrors: DenseNDArray
  ) {

    val softmaxGradients: DenseNDArray = Softmax().df(importanceScore)
    val acErrors: DenseNDArray = softmaxGradients.dot(importanceScoreErrors)

    contextVectorErrors.assignValues(acErrors.t.dot(attentionMatrix.values).t)

    attentionMatrix.assignErrorsByDot(acErrors, contextVector.t)
  }
}
