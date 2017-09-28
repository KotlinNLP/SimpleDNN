/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The optimizer of the [EmbeddingsContainer].
 *
 * @property embeddingsContainer embeddings container to optimize
 * @property updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class EmbeddingsOptimizer(
  val embeddingsContainer: EmbeddingsContainer,
  updateMethod: UpdateMethod<*>
) : Optimizer(updateMethod) {

  /**
   * A support structure to store errors.
   */
  private data class EmbeddingsErrors(val errors: DenseNDArray, var count: Int)

  /**
   * Map an embeddings id with its errors
   */
  private val embeddingsErrorsMap = mutableMapOf<Int, EmbeddingsErrors>()

  /**
   * Update the embeddings.
   */
  override fun update() {

    for ((embeddingIndex, embeddingsErrors) in this.embeddingsErrorsMap) {
      embeddingsErrors.errors.assignDiv(embeddingsErrors.count.toDouble()) // average errors
      this.updateMethod.update(this.embeddingsContainer.getEmbedding(embeddingIndex).array, embeddingsErrors.errors)
    }

    this.embeddingsErrorsMap.clear()
  }

  /**
   * Accumulate errors of the embeddings with the given [embeddingId].
   * If [embeddingId] is null [errors] will be associated to the nullEmbedding.
   * If [embeddingId] is negative or greater then the embeddings count [errors] will be associated to the
   * unknownEmbedding.
   *
   * @param embeddingId id of the embedding on which to accumulate the [errors] (can be null)
   * @param errors errors to accumulate
   */
  fun accumulate(embeddingId: Int?, errors: DenseNDArray) {

    val index: Int = when (embeddingId) {
      null -> -2
      in 0 until this.embeddingsContainer.count -> embeddingId
      else -> -1 // unknown
    }

    val embeddingsErrors: EmbeddingsErrors? = this.embeddingsErrorsMap[index]

    if (embeddingsErrors != null) {
        embeddingsErrors.errors.assignSum(errors)
        embeddingsErrors.count += 1

    } else {
      this.embeddingsErrorsMap[index] = EmbeddingsErrors(errors = errors.copy(), count = 1)
    }
  }
}
