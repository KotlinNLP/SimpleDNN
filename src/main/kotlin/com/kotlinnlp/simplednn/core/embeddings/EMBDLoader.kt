/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.kotlinnlp.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.DictionarySet
import java.io.File
import java.io.InputStreamReader

/**
 * An helper that creates an [EmbeddingsMapByDictionary] associating words to embeddings loaded from a given file.
 * All the words of the embeddings file are automatically added to the dictionary.
 *
 * @param verbose a Boolean indicating whether to enable the verbose mode (default = true)
 */
class EMBDLoader(private val verbose: Boolean = true) {

  /**
   * The name of the loading file containing the embeddings associations.
   */
  private lateinit var filename: String

  /**
   * The progress indicator used in verbose mode.
   */
  private lateinit var progress: ProgressIndicatorBar

  /**
   * The number of embeddings associations in the loading file.
   */
  private var count: Int = 0

  /**
   * The size of the embeddings of the loading file.
   */
  private var embeddingsSize: Int = 0

  /**
   * Create a new [EmbeddingsMapByDictionary], associating words to embeddings loaded from a given file.
   *
   * The file must contain one header line and N following data lines:
   *   - The header line must contain the number N of data lines and the size S of the vectors (the same for all),
   *     separated by a space.
   *   - Each data line must contain the key, followed by S double numbers, each separated by a space.
   *
   * @param filename the input filename
   * @param pseudoRandomDropout the pseudoRandomDropout that is propagated to the [EmbeddingsMapByDictionary]
   *                            constructor (default = true)
   *
   * @return an [EmbeddingsMapByDictionary] with embeddings loaded from the given file
   */
  fun load(filename: String, pseudoRandomDropout: Boolean = true): EmbeddingsMapByDictionary {

    this.initialize(filename = filename)

    val embeddingsMap = EmbeddingsMapByDictionary(
      size = this.embeddingsSize,
      dictionary = DictionarySet(),
      pseudoRandomDropout = pseudoRandomDropout)

    this.forEachDataLine { word, vector ->

      embeddingsMap.dictionary.add(word)

      embeddingsMap.set(
        key = embeddingsMap.dictionary.getId(word)!!,
        embedding = this.buildEmbedding(id = embeddingsMap.count, vector = vector))
    }

    this.checkEmbeddingsCount(embeddingsMap)

    return embeddingsMap
  }

  /**
   * Initialize information of the currently loading file.
   *
   * @param filename the name of the currently loading file
   */
  private fun initialize(filename: String) {

    val firstLine: String = this.readFirstLine(filename)
    val firstLineSplit: List<String> = firstLine.split(" ")

    this.filename = filename
    this.count = firstLineSplit[0].toInt()
    this.embeddingsSize = firstLineSplit[1].toInt()

    if (this.verbose) {
      this.progress = ProgressIndicatorBar(total = File(filename).getNumOfLines() - 1)
    }
  }

  /**
   * Build a new [Embedding] with the given [id] and [vector].
   *
   * @param id the id of the embeddings
   * @param vector the vector of the embedding
   *
   * @return a new embedding
   */
  private fun buildEmbedding(id: Int, vector: DoubleArray) = Embedding(
    id = id,
    array = UpdatableDenseArray(values = DenseNDArrayFactory.arrayOf(vector)))

  /**
   * Check if the given [embeddingsMap] contains the expected number of embeddings.
   *
   * @throws IllegalArgumentException if the number of embeddings in the given [embeddingsMap] differs from the amount
   *                                  written in the header of the currently loading file
   */
  private fun checkEmbeddingsCount(embeddingsMap: EmbeddingsMapByDictionary) {
    require(embeddingsMap.count == this.count) {
      "Invalid file: wrong number of declared embeddings (%d != %d).".format(embeddingsMap.count, this.count)
    }
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
   * Loop the data lines of the current file.
   *
   * @param callback the callback called for each line, passing it the word and the vector of the line
   */
  private fun forEachDataLine(callback: (word: String, vector: DoubleArray) -> Unit) {

    var isFirstLine = true

    File(this.filename).forEachLine { line ->

      if (isFirstLine) {
        isFirstLine = false

      } else {

        val elements: List<String> = line.split(" ")
        val vector = DoubleArray(size = this.embeddingsSize, init = { i -> elements[i + 1].toDouble() })

        if (this.verbose) {
          this.progress.tick()
        }

        callback(elements.first(), vector)
      }
    }
  }

  /**
   *
   */
  private fun File.getNumOfLines(): Int {

    var numOfLines = 0

    this.reader().forEachLine { numOfLines++ }

    return numOfLines
  }
}
