/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The optimizer of the [EmbeddingsMap].
 *
 * @param embeddingsMap the [EmbeddingsMap] to optimize
 * @param updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class EmbeddingsOptimizer<in T>(
  private val embeddingsMap: EmbeddingsMap<T>,
  updateMethod: UpdateMethod<*>
) : Optimizer<Pair<Int, DenseNDArray>>(updateMethod) {

  /**
   * A support data class to store errors and accumulations count.
   */
  private data class ErrorsAccumulator(val errors: DenseNDArray, var count: Int) {

    fun accumulate(errors: DenseNDArray) {
      this.errors.assignSum(errors)
      this.count += 1
    }
  }

  /**
   * Map embeddings embeddings ids to their errors.
   */
  private val embeddingsErrorsMap = mutableMapOf<Int, ErrorsAccumulator>()

  /**
   * Update the embeddings.
   */
  override fun update() {

    for ((id, embeddingErrors) in this.embeddingsErrorsMap) {

      embeddingErrors.errors.assignDiv(embeddingErrors.count.toDouble()) // average errors

      this.updateMethod.update(this.embeddingsMap.getById(id)!!.array, embeddingErrors.errors)
    }

    this.embeddingsErrorsMap.clear()
  }

  /**
   * Accumulate the errors of the embedding identified with its Id in the [embeddingsMap].
   *
   * @param paramsErrors the pair of id and errors
   */
  override fun accumulate(paramsErrors: Pair<Int, DenseNDArray>, copy: Boolean) {

    val item = this.embeddingsErrorsMap[paramsErrors.first]

    if (item != null) {
      item.accumulate(paramsErrors.second)
    } else {
      this.embeddingsErrorsMap[paramsErrors.first] = ErrorsAccumulator(errors = paramsErrors.second.copy(), count = 1)
    }
  }

  /**
   * Accumulate the [errors] of the given [embedding].
   *
   * @param embedding the embedding on which to accumulate the [errors]
   * @param errors errors to accumulate
   */
  fun accumulate(embedding: Embedding, errors: DenseNDArray) {
    this.accumulate(Pair(embedding.id, errors))
  }

  /**
   * Accumulate errors of the embeddings with the given [embeddingKey].
   * If [embeddingKey] is null [errors] will be associated to the nullEmbedding.
   * If [embeddingKey] is negative or greater then the embeddings count [errors] will be associated to the
   * unknownEmbedding.
   *
   * @param embeddingKey the key of the embedding on which to accumulate the [errors] (can be null)
   * @param errors errors to accumulate
   */
  fun accumulate(embeddingKey: T?, errors: DenseNDArray) {
    this.accumulate(Pair(this.embeddingsMap.get(embeddingKey).id, errors))
  }
}
