/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings.lmdb

import com.github.matteogrella.lmdbkt.LMDBMap
import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.utils.Serializer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.ByteArrayOutputStream
import java.io.File
import java.lang.RuntimeException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/**
 * Fast word embedding lookup with reduced memory footprint using the Lightning Memory-Mapped Database (LMDB).
 *
 * @param filename path to the database
 * @param readOnly whether to use the resource in read-only mode
 */
class EmbeddingsStorage(
  filename: Path,
  readOnly: Boolean
) : LMDBMap<String, ParamsArray>(
  path = filename,
  readOnly = readOnly
) {

  /**
   * @param filename path to the database
   * @param readOnly whether to use the resource in read-only mode
   */
  constructor(filename: String, readOnly: Boolean) :
    this(filename = Paths.get(filename), readOnly = readOnly)

  /**
   * Exception raised by setter of the [embeddingsSize].
   */
  class ValueAlreadySet : RuntimeException()

  /**
   * The size of each embedding.
   */
  var embeddingsSize: Int = 0
    set(value) { if (field != 0) { throw ValueAlreadySet() }; field = value }
    get() { require(field > 0) { "The value must be set before accessing it" }; return field }

  override fun deserializeKey(obj: ByteArray): String = String(obj)
  override fun deserializeValue(obj: ByteArray): ParamsArray = Serializer.deserialize(obj.inputStream())
  override fun serializeKey(obj: String): ByteArray = obj.toByteArray()
  override fun serializeValue(obj: ParamsArray): ByteArray {
    val outputStream = ByteArrayOutputStream()
    Serializer.serialize(obj, outputStream)
    return outputStream.toByteArray()
  }

  companion object {

    /**
     * @param filename the filename
     *
     * @return the number of lines
     */
    private fun getNumOfLines(filename: String): Int = Files.lines(Paths.get(filename)).count().toInt()

    /**
     * Loop the data lines of the current file.
     *
     * @param callback the callback called for each line, passing it the word and the vector
     */
    private fun forEachDataLine(filename: String, callback: (word: String, vector: DoubleArray) -> Unit) {

      var isFirstLine = true

      File(filename).forEachLine { line ->

        if (isFirstLine) {

          isFirstLine = false

        } else {

          val word = line.substringBefore(' ')
          val strVector = line.substringAfter(' ')
          val vector = strVector.split(" ").let { DoubleArray(size = it.size, init = { i -> it[i].toDouble() }) }

          callback(word, vector)
        }
      }
    }
  }

  /**
   * Load the embeddings from file.
   *
   * The file must contain one header line and N following data lines:
   *   - The header line must contain the number N of data lines and the size S of the vectors (the same for all),
   *     separated by a space.
   *   - Each data line must contain the key, followed by S double numbers, each separated by a space.
   *
   * @param filename the filename of the embeddings to load
   * @param overwrite overwrite the vectors if already existing
   * @param verbose a Boolean indicating whether to enable the verbose mode (default = true)
   */
  fun load(filename: String, overwrite: Boolean = true, verbose: Boolean = true) {

    if (overwrite) this.clear()

    val progress: ProgressIndicatorBar? =
      if (verbose)
        ProgressIndicatorBar(total = getNumOfLines(filename) - 1)
      else
        null

    var firstInsertion = true

    forEachDataLine(filename) { word, vector ->

      if (overwrite || word !in this) {

        if (firstInsertion) {

          try {
            this.embeddingsSize = vector.size
            if (verbose) println("Set the embeddings size to ${vector.size}")
          } catch (e: ValueAlreadySet) {
            if (verbose) println("The size has been already set to ${this.embeddingsSize}")
          }

          firstInsertion = false
        }

        require(vector.size == this.embeddingsSize) {
          "The embeddings must have the same length. Found ${vector.size}, expected ${this.embeddingsSize}."
        }

        this[word] = ParamsArray(vector)
      }

      progress?.tick()
    }
  }
}
