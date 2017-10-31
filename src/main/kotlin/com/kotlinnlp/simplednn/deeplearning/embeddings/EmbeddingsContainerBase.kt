/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import java.io.Serializable
import java.util.*

/**
 * A container of Embeddings mapped to Int ids.
 *
 * @property count the number of embeddings in this [EmbeddingsContainerBase] (e.g. number of word in a vocabulary)
 * @property size the size of each embedding (typically a range between about 50 to a few hundreds)
 * @property pseudoRandomDropout a Boolean indicating if Embeddings must be dropped out with pseudo random probability
 */
abstract class EmbeddingsContainerBase<SelfType: EmbeddingsContainerBase<SelfType>>(
  val count: Int,
  val size: Int,
  private val pseudoRandomDropout: Boolean = true
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The Unknown Embedding.
   */
  val unknownEmbedding = this.buildEmbedding(id = -1)

  /**
   * The Null Embedding.
   */
  val nullEmbedding = this.buildEmbedding(id = -2)

  /**
   * Map embeddings ids to vectors, i.e. id 7 to vector [0.18, 0.12, 0.87...].
   */
  private val embeddings = Array(size = count, init = { index -> this.buildEmbedding(index) })

  /**
   * The random generator used to decide if an Embedding must be dropped out.
   */
  private val randomGenerator = if (this.pseudoRandomDropout) Random(743) else Random()

  /**
   * Get the embedding with the given [id] as Int.
   * If the [id] is null return the [nullEmbedding].
   * If the [id] is negative or greater than [count] return the [unknownEmbedding].
   *
   * @param id (can be null)
   * @param dropout the probability to get the [unknownEmbedding] (default = 0.0 = no dropout)
   *
   * @return the [Embedding] with the given [id] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getEmbedding(id: Int?, dropout: Double = 0.0): Embedding {
    require(dropout in 0.0..1.0)

    return when {
      dropout > 0.0 && this.mustBeDropped(dropout) -> this.unknownEmbedding
      id != null -> if (id in 0 until this.count) this.embeddings[id] else this.unknownEmbedding
      else -> this.nullEmbedding
    }
  }

  /**
   * Random embeddings initialization.
   *
   * @param randomGenerator a [RandomGenerator]
   *
   * @return this [EmbeddingsContainerBase]
   */
  fun initialize(randomGenerator: RandomGenerator = FixedRangeRandom(radius = 0.08, enablePseudoRandom = true))
    : SelfType {

    this.embeddings.forEach { it.array.values.randomize(randomGenerator) }

    this.nullEmbedding.array.values.randomize(randomGenerator)
    this.unknownEmbedding.array.values.randomize(randomGenerator)

    @Suppress("UNCHECKED_CAST")
    return this as SelfType
  }

  /**
   * @param id the id associated to the [Embedding] to build
   *
   * @return a new [Embedding] with the given [id]
   */
  private fun buildEmbedding(id: Int) = Embedding(id = id, array = UpdatableDenseArray(Shape(this.size)))

  /**
   * @param dropout the probability of dropout
   *
   * @return a Boolean indicating if an Embedding must be dropped out
   */
  private fun mustBeDropped(dropout: Double): Boolean {
    return this.randomGenerator.nextDouble() < dropout
  }
}
