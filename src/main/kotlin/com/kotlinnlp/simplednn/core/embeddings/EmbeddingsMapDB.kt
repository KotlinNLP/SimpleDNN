/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings

import com.github.shyiko.levelkt.LevelDBMap
import com.kotlinnlp.utils.Serializer
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.file.Paths

/**
 * @param filename the filename of the data-base
 * @param size the size of each embedding (typically a range between about 50 to a few hundreds)
 * @param pseudoRandomDropout a Boolean indicating if Embeddings must be dropped out with pseudo random probability
 */
class EmbeddingsMapDB(
  filename: String,
  size: Int,
  pseudoRandomDropout: Boolean = true
) : EmbeddingsMap<String>(
  size = size,
  pseudoRandomDropout = pseudoRandomDropout
), Closeable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Transform an Integer to a Byte Array.
     *
     * @param value the Integer
     *
     * @return the Byte Array
     */
    private fun getByteArrayFromInt(value: Int): ByteArray {

      val mask = 0xFF // binary 1111 1111
      var number = value
      val result = ByteArray(java.lang.Integer.BYTES) { 0 }

      for (i in 0 until result.size) {
        result[i] = number.and(mask).toByte()
        number = number.shr(8)
      }

      result.reverse()

      return result
    }

    /**
     * Transform an Embedding to a Byte Array.
     *
     * @param value the Embedding
     *
     * @return the Byte Array
     */
    private fun getByteArrayFromEmbedding(value: Embedding): ByteArray {

      val outputStream = ByteArrayOutputStream()
      Serializer.serialize(value, outputStream)
      return outputStream.toByteArray()
    }
  }

  /**
   * The map of keys to embeddings.
   */
  override val embeddings = object : LevelDBMap<String, Embedding>(Paths.get(filename)) {
    override fun deserializeKey(obj: ByteArray): String = String(obj)
    override fun deserializeValue(obj: ByteArray): Embedding = Serializer.deserialize(obj.inputStream())
    override fun serializeKey(obj: String): ByteArray = obj.toByteArray()
    override fun serializeValue(obj: Embedding): ByteArray = getByteArrayFromEmbedding(obj)
  }

  /**
   * The map of ids to embeddings.
   */
  override val embeddingsById = object : LevelDBMap<Int, Embedding>(Paths.get("$filename-id")) {
    override fun deserializeKey(obj: ByteArray): Int = ByteBuffer.wrap(obj).int
    override fun deserializeValue(obj: ByteArray): Embedding = Serializer.deserialize(obj.inputStream())
    override fun serializeKey(obj: Int): ByteArray = getByteArrayFromInt(obj)
    override fun serializeValue(obj: Embedding): ByteArray = getByteArrayFromEmbedding(obj)
  }

  /**
   * Initialize the map with the 'unknown' and 'null' embeddings.
   */
  init {
    this.embeddingsById.putAll(mapOf(
      this.unknownEmbedding.id to this.unknownEmbedding,
      this.nullEmbedding.id to this.nullEmbedding
    ))
  }

  /**
   * Release the resources.
   */
  override fun close() {

    this.embeddings.close()
    this.embeddingsById.close()
  }
}
