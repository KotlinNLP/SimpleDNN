/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator

/**
 * The EmbeddingsContainer
 *
 * @property count the number of embeddings of this EmbeddingsContainer (e.g. number of word in a vocabulary=
 * @property size the size of the embeddings (typically a range between about 50 to a few hundreds)
 */
class EmbeddingsContainer(val count: Int, val size: Int) {

  /**
   * An Embedding is a dense vectors of real numbers.
   *
   * @property index the index of the Embedding in the lookupTable
   * @property array the values of the Embedding
   */
  data class Embedding(val index: Int, val array: UpdatableArray)

  /**
   * Out-of-vocabulary embeddings used to represent unknown-item
   */
  val unknownEmbeddingsId: Int = this.count

  /**
   * Out-of-vocabulary embeddings used to represent null-item
   */
  val nullEmbeddingsId: Int = this.count + 1
  
  /**
   * Map from indices to vectors, i.e. index 7 to vector [0.18, 0.12, 0.87...].
   * The last two elements of the look-up table are the 'unknown' and 'null' vectors
   */
  val lookupTable = Array(
    size = count + 2,
    init = { Embedding(index = it, array = UpdatableArray(length = this.size)) }
  )

  /**
   * Unknown Embedding
   */
  val unknownEmbedding get() = this.lookupTable[this.unknownEmbeddingsId]

  /**
   * Null Embedding
   */
  val nullEmbedding get() = this.lookupTable[this.nullEmbeddingsId]

  /**
   * Return the embedding at the given index.
   * If the index is null return the nullEmbedding.
   *
   * @param index (can be null)
   *
   * @return the embedding at the given index
   */
  fun getEmbedding(index: Int?): Embedding {
    require(index == null || index in 0 .. this.lookupTable.size)
    return this.lookupTable[index ?: this.nullEmbeddingsId]
  }

  /**
   * Random embeddings initialization.
   *
   * @param randomGenerator
   *
   * @return this EmbeddingContainer
   */
  fun randomize(randomGenerator: RandomGenerator = FixedRangeRandom(radius = 0.08, enablePseudoRandom = true)): EmbeddingsContainer {
    this.lookupTable.forEach { it.array.values.randomize(randomGenerator) }
    return this
  }
}
