/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.embeddings

import com.kotlinnlp.simplednn.utils.DictionarySet
import java.io.File

/**
 * An [EmbeddingsMap] that get embeddings eventually with a probability of dropout related to their elements frequency
 * in the given [dictionary].
 * Only embeddings associated to elements contained in the [dictionary] can be get, otherwise the [unknownEmbedding] is
 * returned.
 *
 * @property dictionary a dictionary set
 * @param size the size of each embedding (typically a range between about 50 to a few hundreds)
 * @param pseudoRandomDropout a Boolean indicating if Embeddings must be dropped out with pseudo random probability
 */
class EmbeddingsMapByDictionary(
  val dictionary: DictionarySet<String>,
  size: Int,
  pseudoRandomDropout: Boolean = true
) : EmbeddingsMap<Int>(
  size = size,
  pseudoRandomDropout = pseudoRandomDropout) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Get the embedding associated to the given [element], eventually with a dropout probability.
   * If the [element] is null return the [nullEmbedding].
   * If the [element] is not in the dictionary or it is dropped return the [unknownEmbedding].
   *
   * @param element a string element (can be null)
   * @param dropoutCoefficient the dropout coefficient
   *
   * @return the [Embedding] with associated to the given [element] or [nullEmbedding] or [unknownEmbedding]
   */
  fun get(element: String?, dropoutCoefficient: Double = 0.0): Embedding {
    require(dropoutCoefficient in 0.0 .. 1.0)

    return when (element) {
      null -> this.nullEmbedding
      in this.dictionary -> {
        val id: Int = this.dictionary.getId(element)!!
        this.getOrSet(key = id, dropout = this.getElementDropout(id = id, dropoutCoefficient = dropoutCoefficient))
      }
      else -> this.unknownEmbedding
    }
  }

  /**
   * Get the dropout probability of the element with the given [id], in relation to its frequency in the [dictionary].
   *
   * @param id the id of an element in the dictionary
   * @param dropoutCoefficient the dropout coefficient
   *
   * @return the probability to get the [unknownEmbedding]
   */
  private fun getElementDropout(id: Int, dropoutCoefficient: Double): Double =
    if (dropoutCoefficient > 0.0)
      dropoutCoefficient / (this.dictionary.getCount(id) + dropoutCoefficient)
    else
      0.0

  /**
   * Export the embeddings map writing its entries to a file with the given [filename].
   *
   * The file will contain one header line and N following data lines:
   *   - The header line contains the number N of data lines and the size S of the vectors (the same for all),
   *     separated by a space.
   *   - Each data line contains the key, followed by S double numbers, each separated by a space.
   *
   * @param filename the output filename
   * @param digits precision specifier
   */
  fun dump(filename: String, digits: Int) {

    File(filename).printWriter().use { out ->

      out.println("%d %d".format(this.count, this.size))

      this.dictionary.getElements().forEach {
        out.print(it)
        out.println(this.get(it).toString(digits = digits))
        out.flush()
      }
    }
  }
}
