/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import java.util.*

/**
 * The EmbeddingsHelper is mainly a simple wrapper of the EmbeddingsContainer.
 * It adds dropout functionality and include the optimizer.
 *
 * @property embeddings the embeddings container
 * @property updateMethod the [UpdateMethod] used to update the [embeddings] during the optimization
 */
class EmbeddingsHelper(
  val embeddings: EmbeddingsContainer,
  val updateMethod: UpdateMethod,
  val dropoutConfiguration: DropoutConfiguration): Optimizer {

  /**
   * Dropout configuration
   *
   * @property alpha dropout constant
   * @property enablePseudoRandom whether to enable pseudo random
   * @property seed the seed used for the pseudo random
   */
  data class DropoutConfiguration(
    val alpha: Double = 0.0,
    val enablePseudoRandom: Boolean = true,
    val seed: Long = 1)

  /**
   * Random generator used by [getEmbeddingDropout]
   */
  private val rndGenerator = if (this.dropoutConfiguration.enablePseudoRandom) {
    Random(this.dropoutConfiguration.seed)
  } else {
    Random()
  }

  /**
   * The optimizer of the [embeddings]
   */
  private val optimizer = EmbeddingsOptimizer(
    embeddingsContainer = this.embeddings,
    updateMethod = this.updateMethod)

  /**
   * Replace an embedding with the 'unknown' with probability
   * that is inversely proportional to the [occurrences]
   */
  fun getEmbeddingDropout(index: Int?, occurrences: Int): EmbeddingsContainer.Embedding =
    if (rndGenerator.nextDouble() < this.dropoutConfiguration.alpha / (occurrences + this.dropoutConfiguration.alpha)){
      this.embeddings.unknownEmbedding
    } else {
      this.embeddings.getEmbedding(index)
    }

  /**
   * Return the embedding at the given index.
   * If the index is null return the nullEmbedding.
   *
   * @param index (can be null)
   *
   * @return the embedding at the given index
   */
  fun getEmbedding(index: Int?): EmbeddingsContainer.Embedding = this.embeddings.getEmbedding(index)

  /**
   * @return the 'unknown' embedding
   */
  fun getUnknownEmbedding() = this.embeddings.unknownEmbedding

  /**
   * Propagate the errors on the embedding
   *
   * @param embedding embedding to which propagate the [outputErrors]
   * @param outputErrors the errors to propagate on the [embedding]
   */
  fun propagateErrors(embedding: EmbeddingsContainer.Embedding, outputErrors: DenseNDArray) {
    optimizer.accumulateErrors(embeddingIndex = embedding.index, errors = outputErrors)
  }

  /**
   * In turn it calls the same method into the `optimizer`
   */
  override fun update() {
    this.optimizer.update()
  }

  /**
   * Method to call every new epoch.
   *
   * In turn it calls the same method into the `optimizer`
   */
  override fun newEpoch() {
    this.optimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   *
   * In turn it calls the same method into the `optimizer`
   */
  override fun newBatch() {
    this.optimizer.newBatch()
  }

  /**
   * Method to call every new example.
   *
   * In turn it calls the same method into the `optimizer`
   */
  override fun newExample() {
    this.optimizer.newExample()
  }
}
