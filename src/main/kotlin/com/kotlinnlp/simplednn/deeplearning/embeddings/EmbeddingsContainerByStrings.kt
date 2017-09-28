/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

/**
 * An extension of the [EmbeddingsContainer] which can also map Embeddings also to strings.
 *
 * @property count the number of embeddings in this [EmbeddingsContainer] (e.g. number of word in a vocabulary=
 * @property size the size of each embedding (typically a range between about 50 to a few hundreds)
 */
class EmbeddingsContainerByStrings(count: Int, size: Int) : EmbeddingsContainer(count = count, size = size) {

  /**
   * Map strings to ids of embeddings.
   */
  private val idsMap = mutableMapOf<String, Int>()

  /**
   * Get the embedding with the given [id] as String.
   * If the [id] is null return the [nullEmbedding].
   * If the [id] is negative or greater than [count] return the [unknownEmbedding].
   *
   * @param id (can be null)
   *
   * @return the Embedding with the given [id] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getEmbedding(id: String?): Embedding {

    return if (id == null) {

      super.getEmbedding(id = null)

    } else {

      if (!this.idsMap.containsKey(id)) {
        this.idsMap[id] = this.idsMap.size
      }

      super.getEmbedding(this.idsMap[id]!!)
    }
  }
}
