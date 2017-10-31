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
 * The optimizer of the [EmbeddingsMap].
 *
 * @property embeddingsMap the [EmbeddingsMap] to optimize
 * @property updateMethod the [UpdateMethod] for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class EmbeddingsOptimizer<in T>(
  val embeddingsMap: EmbeddingsMap<T>,
  updateMethod: UpdateMethod<*>
) : Optimizer(updateMethod) {

  /**
   * A support data class to store errors and accumulations cou.
   */
  private data class EmbeddingsErrors(val errors: DenseNDArray, var count: Int)

  /**
   * The errors associated to the nullEmbedding.
   */
  private var nullEmbeddingErrors: EmbeddingsErrors? = null

  /**
   * The errors associated to the unknownEmbedding.
   */
  private var unknownEmbeddingErrors: EmbeddingsErrors? = null

  /**
   * Map embeddings keys to their errors.
   */
  private val embeddingsErrorsMap = mutableMapOf<T, EmbeddingsErrors>()

  /**
   * Update the embeddings.
   */
  override fun update() {

    for ((key, embeddingErrors) in this.embeddingsErrorsMap) {
      this.updateEmbedding(embedding = this.embeddingsMap.get(key), errors = embeddingErrors)
    }

    if (this.nullEmbeddingErrors != null) {
      this.updateEmbedding(embedding = this.embeddingsMap.nullEmbedding, errors = this.nullEmbeddingErrors!!)
    }

    if (this.unknownEmbeddingErrors != null) {
      this.updateEmbedding(embedding = this.embeddingsMap.unknownEmbedding, errors = this.unknownEmbeddingErrors!!)
    }

    this.embeddingsErrorsMap.clear()
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

    when (embeddingKey) {

      null -> this.accumulateNullEmbeddingErrors(errors) // null

      !in this.embeddingsMap -> this.accumulateUnknownEmbeddingErrors(errors) // unknown

      else -> { // other

        if (embeddingKey in this.embeddingsErrorsMap) {
          val embeddingErrors: EmbeddingsErrors = this.embeddingsErrorsMap[embeddingKey]!!
          embeddingErrors.errors.assignSum(errors)
          embeddingErrors.count += 1

        } else {
          this.embeddingsErrorsMap[embeddingKey] = EmbeddingsErrors(errors = errors.copy(), count = 1)
        }
      }
    }
  }

  /**
   * Accumulate errors of the nullEmbedding.
   *
   * @param errors the errors to accumulate
   */
  private fun accumulateNullEmbeddingErrors(errors: DenseNDArray) {

    if (this.nullEmbeddingErrors != null) {
      this.nullEmbeddingErrors!!.errors.assignSum(errors)
      this.nullEmbeddingErrors!!.count += 1

    } else {
      this.nullEmbeddingErrors = EmbeddingsErrors(errors = errors.copy(), count = 1)
    }
  }

  /**
   * Accumulate errors of the unknownEmbedding.
   *
   * @param errors the errors to accumulate
   */
  private fun accumulateUnknownEmbeddingErrors(errors: DenseNDArray) {

    if (this.unknownEmbeddingErrors != null) {
      this.unknownEmbeddingErrors!!.errors.assignSum(errors)
      this.unknownEmbeddingErrors!!.count += 1

    } else {
      this.unknownEmbeddingErrors = EmbeddingsErrors(errors = errors.copy(), count = 1)
    }
  }

  /**
   * Update an [embedding] given its [errors].
   *
   * @param embedding the embedding to update
   * @param errors the embedding errors object
   */
  private fun updateEmbedding(embedding: Embedding, errors: EmbeddingsErrors) {

    errors.errors.assignDiv(errors.count.toDouble()) // average errors

    this.updateMethod.update(embedding.array, errors.errors)
  }
}
