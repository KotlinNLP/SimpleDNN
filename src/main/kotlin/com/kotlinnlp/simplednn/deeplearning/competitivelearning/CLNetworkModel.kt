/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning

import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of a [CLNetwork].
 *
 * @property numOfClasses the number of classes
 */
abstract class CLNetworkModel(val numOfClasses: Int): Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [CLNetworkModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [CLNetworkModel]
     *
     * @return the [CLNetworkModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): CLNetworkModel = Serializer.deserialize(inputStream)
  }

  /**
   * The range of classes.
   */
  val classes = IntRange(0, this.numOfClasses - 1)

  /**
   * Serialize this [CLNetworkModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [CLNetworkModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
