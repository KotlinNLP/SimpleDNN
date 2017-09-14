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

/**
 * The EmbeddingsContainer
 *
 * @property count the number of embeddings in this [EmbeddingsContainer] (e.g. number of word in a vocabulary=
 * @property size the size of each embedding (typically a range between about 50 to a few hundreds)
 */
class EmbeddingsContainer(val count: Int, val size: Int) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * An Embedding is a dense vectors of real numbers.
   *
   * @property index the index of the Embedding in the lookupTable
   * @property array the values of the Embedding
   */
  data class Embedding(val index: Int, val array: UpdatableDenseArray) : Serializable {

    companion object {

      /**
       * Private val used to serialize the class (needed from Serializable)
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L
    }
  }
  
  /**
   * Map from indices to vectors, i.e. index 7 to vector [0.18, 0.12, 0.87...].
   * The last two elements of the look-up table are the 'unknown' and 'null' vectors.
   */
  private val lookupTable = Array(size = count, init = { index -> this.buildEmbedding(index) })

  /**
   * Map strings to ids of embeddings.
   */
  private val idsMap = mutableMapOf<String, Int>()

  /**
   * The Unknown Embedding.
   */
  val unknownEmbedding = this.buildEmbedding(index = -1)

  /**
   * The Null Embedding.
   */
  val nullEmbedding = this.buildEmbedding(index = -2)

  /**
   * Get the embedding with the given [id] as Int.
   * If the [id] is null return the [nullEmbedding].
   * If the [id] is negative or greater than [count] return the [unknownEmbedding].
   *
   * @param id (can be null)
   *
   * @return the [Embedding] with the given [id] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getEmbeddingByInt(id: Int?): Embedding {

    return if (id != null) {

      if (id in 0 until this.count)
        this.lookupTable[id]

      else
        this.unknownEmbedding

    } else {
      this.nullEmbedding
    }
  }

  /**
   * Get the embedding with the given [id] as String.
   * If the [id] is null return the [nullEmbedding].
   * If the [id] is negative or greater than [count] return the [unknownEmbedding].
   *
   * @param id (can be null)
   *
   * @return the [Embedding] with the given [id] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getEmbeddingByString(id: String?): Embedding {

    return if (id == null) {

      this.getEmbeddingByInt(id = null)

    } else {

      if (!this.idsMap.containsKey(id)) {
        this.idsMap[id] = this.idsMap.size
      }

      this.getEmbeddingByInt(this.idsMap[id]!!)
    }
  }

  /**
   * Random embeddings initialization.
   *
   * @param randomGenerator a [RandomGenerator]
   *
   * @return this [EmbeddingsContainer]
   */
  fun randomize(randomGenerator: RandomGenerator = FixedRangeRandom(radius = 0.08, enablePseudoRandom = true))
    : EmbeddingsContainer {

    this.lookupTable.forEach { it.array.values.randomize(randomGenerator) }

    this.nullEmbedding.array.values.randomize(randomGenerator)
    this.unknownEmbedding.array.values.randomize(randomGenerator)

    return this
  }

  /**
   * @param index the index associated to the [Embedding] to build
   *
   * @return a new [Embedding] with the given [index]
   */
  private fun buildEmbedding(index: Int) = Embedding(index = index, array = UpdatableDenseArray(Shape(this.size)))
}
