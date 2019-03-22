/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of the [SequenceParallelEncoder].
 *
 * @property networks a list of sequence feedforward networks
 */
class ParallelEncoderModel(val networks: List<StackedLayersParameters>) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [ParallelEncoderModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [ParallelEncoderModel]
     *
     * @return the [ParallelEncoderModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): ParallelEncoderModel = Serializer.deserialize(inputStream)
  }

  /**
   * The parameters of all networks.
   */
  val params = this.networks.flatMap { it.paramsList }

  /**
   * Serialize this [ParallelEncoderModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [ParallelEncoderModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
