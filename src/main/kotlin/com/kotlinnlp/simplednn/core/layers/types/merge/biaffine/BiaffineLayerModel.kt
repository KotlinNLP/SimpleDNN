/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.merge.biaffine

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of a Biaffine Layer.
 *
 * @property params the parameters of the [BiaffineLayerStructure]s of the pool
 * @property activationFunction the activation function of the [BiaffineLayerStructure]s of the pool
 */
data class BiaffineLayerModel(
  val params: BiaffineLayerParameters,
  val activationFunction: ActivationFunction
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [BiaffineLayerModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [BiaffineLayerModel]
     *
     * @return the [BiaffineLayerModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): BiaffineLayerModel = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [BiaffineLayerModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BiaffineLayerModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
