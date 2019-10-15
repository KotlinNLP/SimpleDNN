/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.utils.getLinesCount
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
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
     * @param verbose a Boolean indicating whether to enable the verbose mode (default = true)
     *
     * @return an [EmbeddingsMap] of [String]s loaded from the given file
     */
    fun load(filename: String,
             pseudoRandomDropout: Boolean = true,
             initializer: Initializer? = GlorotInitializer(),
             verbose: Boolean = true): EmbeddingsMap<String> {

      val progress: ProgressIndicatorBar? = if (verbose)
        ProgressIndicatorBar(total = getLinesCount(filename) - 1)
      else
        null

      val firstLine: String = readFirstLine(filename)
      val firstLineSplit: List<String> = firstLine.split(" ")

      val count: Int = firstLineSplit[0].toInt()
      val size: Int = firstLineSplit[1].toInt()

      val embeddingsMap = EmbeddingsMap<String>(
        size = size,
        pseudoRandomDropout = pseudoRandomDropout,
        initializer = initializer)

      forEachDataLine(filename) { key, vector ->
        embeddingsMap.set(key = key, embedding = ParamsArray(vector))
        progress?.tick()
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
     * @param callback the callback called for each line, passing it the key and the vector of the line
     */
    private fun forEachDataLine(filename: String,
                                callback: (key: String, vector: DoubleArray) -> Unit) {

      var isFirstLine = true

      File(filename).forEachLine { line ->

        if (isFirstLine) {
          isFirstLine = false

        } else {

          val vector: DoubleArray = line
            .trimEnd() // remove trailing whitespaces
            .substringAfter(' ')
            .split(" ")
            .let { DoubleArray(size = it.size, init = { i -> it[i].toDouble() }) }

          callback(line.substringBefore(' '), vector)
        }
      }
    }

    /**
     * Load an [EmbeddingsMap] with [String] keys from the given [elements] set.
     *
     * @param elements the set of elements from which to create the embedding map
     * @param size the size of each embedding
     * @param pseudoRandomDropout the pseudoRandomDropout that is propagated to the [EmbeddingsMap] constructor
     * @param initializer the initializer of the values of the other embeddings (zeros if null, default: Glorot)
     *
     * @return an [EmbeddingsMap] of [String]s
     */
    fun fromSet(elements: Set<String>,
                size: Int,
                initializer: Initializer? = GlorotInitializer(),
                pseudoRandomDropout: Boolean = true) =
      EmbeddingsMap<String>(
        size = size,
        initializer = initializer,
        pseudoRandomDropout = pseudoRandomDropout
      ).apply {
        elements.forEach { embeddingKey -> set(embeddingKey) }
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

        this.embeddings.forEach { (key, value) ->
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
   * The set of keys.
   */
  val keys: Set<T> get() = this.embeddings.keys.toSet()

  /**
   * The Unknown Embedding.
   */
  val unknownEmbedding: ParamsArray = this.buildEmbedding()

  /**
   * The Null Embedding.
   */
  val nullEmbedding: ParamsArray = this.buildEmbedding()

  /**
   * The map of keys to embeddings.
   */
  protected open val embeddings: MutableMap<T, ParamsArray> = mutableMapOf()

  /**
   * The random generator used to decide if an embedding must be dropped out.
   */
  private val dropoutRandomGenerator = if (this.pseudoRandomDropout) Random(743) else Random()

  /**
   * Associate a new embedding to the given [key].
   * It is required that the [key] is never been associated previously.
   * If [embedding] is null a new randomly initialize [ParamsArray] is associated to the given [key].
   *
   * @param key the key to associate to the new embedding
   * @param embedding the embedding to associate to the given [key] (optional, default = null)
   *
   * @return the embedding set
   */
  fun set(key: T, embedding: ParamsArray? = null): ParamsArray {
    require(key !in this.embeddings) { "Embedding with key '%s' already set.".format(key) }
    require(embedding == null || embedding.values.length == this.size) {
      "Embedding size not compatible (%d != %d).".format(embedding!!.values.length, this.size)
    }

    val newEmbedding: ParamsArray = embedding ?: this.buildEmbedding()

    this.embeddings[key] = newEmbedding

    return newEmbedding
  }

  /**
   * Get the embedding with the given [key].
   * If the [key] is null return the [nullEmbedding].
   * If no embedding has the given [key] return the [unknownEmbedding].
   *
   * @param key the key associated to an embedding (can be null)
   *
   * @return the embedding with the given not-null [key] or [nullEmbedding] or [unknownEmbedding]
   */
  operator fun get(key: T?): ParamsArray = this.get(key = key, dropout = 0.0)

  /**
   * Get the embedding with the given [key].
   * If the [key] is null return the [nullEmbedding].
   * If no embedding has the given [key] return the [unknownEmbedding].
   *
   * @param key the key associated to an embedding (can be null)
   * @param dropout the probability to get the [unknownEmbedding] (0.0 = no dropout)
   *
   * @return the embedding with the given not-null [key] or [nullEmbedding] or [unknownEmbedding]
   */
  fun get(key: T?, dropout: Double): ParamsArray {
    require(dropout in 0.0 .. 1.0)

    return when {
      dropout > 0.0 && this.mustBeDropped(dropout) -> this.unknownEmbedding
      key == null -> this.nullEmbedding
      key in this.embeddings -> this.embeddings.getValue(key)
      else -> this.unknownEmbedding
    }
  }

  /**
   * Get the embedding with the given [key].
   * If the [key] is null return the [nullEmbedding].
   * If no embedding has the given [key] associate a new initialized embedding to it and return it.
   * If dropout > 0.0 and the dropout is applied, return the [unknownEmbedding].
   *
   * @param key the key of an embedding (can be null)
   * @param dropout the probability to get the [unknownEmbedding] (default = 0.0 = no dropout)
   *
   * @return the embedding with the given not-null [key] or [nullEmbedding] or [unknownEmbedding]
   */
  fun getOrSet(key: T?, dropout: Double = 0.0): ParamsArray {
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
   * Build a new embedding with randomly initialized values.
   *
   * @return a new embedding
   */
  private fun buildEmbedding() = ParamsArray(size = this.size, initializer = this.initializer)

  /**
   * @param dropout the probability of dropout
   *
   * @return a Boolean indicating if an Embedding must be dropped out
   */
  private fun mustBeDropped(dropout: Double): Boolean = this.dropoutRandomGenerator.nextDouble() < dropout

  /**
   * @param digits precision specifier
   *
   * @return a string representation of the embedding values, concatenating the elements with the space character.
   */
  private fun ParamsArray.toString(digits: Int): String {

    val sb = StringBuilder()

    (0 until this.values.length).forEach {
      sb.append(" ").append("%.${digits}f".format(this.values[it]))
    }

    return sb.toString()
  }
}
