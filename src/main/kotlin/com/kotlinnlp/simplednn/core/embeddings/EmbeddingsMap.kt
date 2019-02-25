/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import java.io.File
import java.io.InputStreamReader
import java.io.Serializable
import java.util.*

/**
 * A map of generic keys to Embeddings.
 *
 * @param size the size of each embedding
 * @param initializer the initializer of the values of the embeddings (zeros if null, default: Glorot)
 * @param pseudoRandomDropout a Boolean indicating if embeddings must be dropped out with pseudo random probability
 *                            (default = true)
 */
open class EmbeddingsMap<T>(
  val size: Int,
  private val initializer: Initializer? = GlorotInitializer(),
  private val pseudoRandomDropout: Boolean = true
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Load an [EmbeddingsMap] with [String] keys from file.
     *
     * The file must contain one header line and N following data lines:
     *   - The header line must contain the number N of data lines and the size S of the vectors (the same for all),
     *     separated by a space.
     *   - Each data line must contain the key, followed by S double numbers, each separated by a space.
     *
     * @param filename the input filename
     * @param pseudoRandomDropout the pseudoRandomDropout that is propagated to the [EmbeddingsMap] constructor
     * @param initializer the initializer of the values of the other embeddings (zeros if null, default: Glorot)
     *
     * @return an [EmbeddingsMap] of [String]s loaded from the given file
     */
    fun load(filename: String,
             pseudoRandomDropout: Boolean = true,
             initializer: Initializer? = GlorotInitializer()): EmbeddingsMap<String> {

      val firstLine: String = readFirstLine(filename)
      val firstLineSplit: List<String> = firstLine.split(" ")

      val count: Int = firstLineSplit[0].toInt()
      val size: Int = firstLineSplit[1].toInt()

      val embeddingsMap = EmbeddingsMap<String>(
        size = size,
        pseudoRandomDropout = pseudoRandomDropout,
        initializer = initializer)

      forEachDataLine(filename = filename, vectorSize = size) { key, vector ->
        embeddingsMap.set(key = key, embedding = Embedding(id = embeddingsMap.count, vector = vector))
      }

      require(embeddingsMap.count == count) {
        "Invalid file: wrong number of declared embeddings (%d != %d).".format(embeddingsMap.count, count)
      }

      return embeddingsMap
    }

    /**
     * @param filename the name of a file
     *
     * @return the first line of the given file
     */
    private fun readFirstLine(filename: String): String {

      val reader: InputStreamReader = File(filename).reader()
      val firstLine = StringBuffer()
      var char: Char = reader.read().toChar()

      while (char != '\n') {
        firstLine.append(char)
        char = reader.read().toChar()
      }

      return firstLine.toString()
    }

    /**
     * Loop the data lines of the given file.
     *
     * @param filename the input filename
     * @param vectorSize the size of each vector
     * @param callback the callback called for each line, passing it the key and the vector of the line
     */
    private fun forEachDataLine(filename: String,
                                vectorSize: Int,
                                callback: (key: String, vector: DoubleArray) -> Unit) {

      var isFirstLine = true

      File(filename).forEachLine { line ->

        if (isFirstLine) {
          isFirstLine = false

        } else {
          val elements: List<String> = line.split(" ")
          val vector = DoubleArray(size = vectorSize, init = { i -> elements[i + 1].toDouble() })

          callback(elements.first(), vector)
        }
      }
    }

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
    fun EmbeddingsMap<String>.dump(filename: String, digits: Int) {

      File(filename).printWriter().use { out ->

        out.println("%d %d".format(this.count, this.size))

        this.embeddings.forEach { key, value ->
          out.print(key)
          out.println(value.toString(digits = digits))
          out.flush()
        }
      }
    }
  }

  /**
   * The number of embeddings in this [EmbeddingsMap] (excluding the [unknownEmbedding] and the [nullEmbedding]).
   */
  val count: Int get() = this.embeddings.size

  /**
   * The Unknown Embedding.
   */
  val unknownEmbedding: Embedding = this.buildEmbedding(id = -1)

  /**
   * The Null Embedding.
   */
  val nullEmbedding: Embedding = this.buildEmbedding(id = -2)

  /**
   * The map of keys to embeddings.
   */
  protected open val embeddings: MutableMap<T, Embedding> = mutableMapOf()

  /**
   * The map of ids to embeddings.
   */
  protected open val embeddingsById: MutableMap<Int, Embedding> = mutableMapOf(
    Pair(this.unknownEmbedding.id, this.unknownEmbedding),
    Pair(this.nullEmbedding.id, this.nullEmbedding)
  )

  /**
   * The random generator used to decide if an embedding must be dropped out.
   */
  private val dropoutRandomGenerator = if (this.pseudoRandomDropout) Random(743) else Random()

  /**
   * Associate a new embedding to the given [key].
   * It is required that the [key] is never been associated previously.
   * If [embedding] is null a new randomly initialize [Embedding] is associated to the given [key].
   *
   * @param key the key to associate to the new embedding
   * @param embedding the embedding to associate to the given [key] (optional, default = null)
   *
   * @return the [Embedding] set
   */
  fun set(key: T, embedding: Embedding? = null): Embedding {
    require(key !in this.embeddings) { "Embedding with key %s already set.".format(key) }
    require(embedding == null || embedding.array.values.length == this.size) {
      "Embedding size not compatible (%d != %d).".format(embedding!!.array.values.length, this.size)
    }

    val newEmbedding: Embedding = embedding ?: this.buildEmbedding(id = this.count)

    this.embeddings[key] = newEmbedding
    this.embeddingsById[newEmbedding.id] = newEmbedding

    return newEmbedding
  }

  /**
   * Get the embedding with the given [key].
   * If the [key] is null return the [nullEmbedding].
   * If no embedding has the given [key] return the [unknownEmbedding].
   *
   * @param key the key associated to an embedding (can be null)
   * @param dropout the probability to get the [unknownEmbedding] (default = 0.0 = no dropout)
   *
   * @return the [Embedding] with the given not-null [key] or [nullEmbedding] or [unknownEmbedding]
   */
  operator fun get(key: T?, dropout: Double = 0.0): Embedding {
    require(dropout in 0.0 .. 1.0)

    return when {
      dropout > 0.0 && this.mustBeDropped(dropout) -> this.unknownEmbedding
      key == null -> this.nullEmbedding
      key in this.embeddings -> this.embeddings.getValue(key)
      else -> this.unknownEmbedding
    }
  }

  /**
   * Get the embedding with the given [id].
   *
   * @param id the id of an embedding
   *
   * @return the [Embedding] with the given [id] (including the [nullEmbedding] and the [unknownEmbedding])
   */
  fun getById(id: Int): Embedding? = this.embeddingsById[id]

  /**
   * Get the embedding with the given [key].
   * If the [key] is null return the [nullEmbedding].
   * If no embedding has the given [key] associate a new initialized embedding to it and return it.
   * If dropout > 0.0 and the dropout is applied, return the [unknownEmbedding].
   *
   * @param key the key of an embedding (can be null)
   * @param dropout the probability to get the [unknownEmbedding] (default = 0.0 = no dropout)
   *
   * @return the [Embedding] with the given not-null [key] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getOrSet(key: T?, dropout: Double = 0.0): Embedding {
    require(dropout in 0.0 .. 1.0)

    return when {
      dropout > 0.0 && this.mustBeDropped(dropout) -> this.unknownEmbedding
      key != null -> if (key in this.embeddings) this.embeddings[key]!! else this.set(key)
      else -> this.nullEmbedding
    }
  }

  /**
   * @param key a key
   *
   * @return a [Boolean] indicating if the key is already associated to an embedding
   */
  operator fun contains(key: T): Boolean = key in this.embeddings

  /**
   * Build a new [Embedding] with randomly initialized values.
   *
   * @param id the Int id of the [Embedding] to build
   *
   * @return a new [Embedding] with the given [id]
   */
  private fun buildEmbedding(id: Int) = Embedding(id = id, size = this.size, initializer = this.initializer)

  /**
   * @param dropout the probability of dropout
   *
   * @return a Boolean indicating if an Embedding must be dropped out
   */
  private fun mustBeDropped(dropout: Double): Boolean = this.dropoutRandomGenerator.nextDouble() < dropout
}
