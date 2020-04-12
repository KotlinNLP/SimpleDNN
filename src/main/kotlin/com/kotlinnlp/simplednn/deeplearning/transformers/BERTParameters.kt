/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayerParameters
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The BERT parameters.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @property hiddenSize the size of the hidden arrays, after the attention
 * @property multiHeadStack the number of scaled-dot attention layers
 * @property dropout the probability of attention dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTParameters(
  val inputSize: Int,
  val attentionSize: Int,
  val hiddenSize: Int,
  val multiHeadStack: Int,
  val dropout: Double = 0.0,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read [BERTParameters] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [BERTParameters]
     *
     * @return the [BERTParameters] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): BERTParameters = Serializer.deserialize(inputStream)
  }

  /**
   * The parameters of the scaled-dot attention layer.
   */
  val attention = ScaledDotAttentionLayerParameters(
    inputSize = this.inputSize,
    attentionSize = this.attentionSize,
    outputSize = this.hiddenSize,
    weightsInitializer = weightsInitializer)

  /**
   * The parameters of the merge layer of the multi-head attention outputs.
   */
  val multiHeadMerge = ConcatFFLayerParameters(
    inputsSize = List(size = this.multiHeadStack, init = { this.hiddenSize }),
    outputSize = this.inputSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the output feed-forward layer.
   */
  val outputFF = FeedforwardLayerParameters(
    inputSize = this.inputSize,
    outputSize = this.inputSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The sum layer parameters.
   */
  val sum = SumLayerParameters(inputSize = this.inputSize, nInputs = 2)

  /**
   * The updatable normalization scalar parameter.
   */
  var normScalarParam = ParamsArray(doubleArrayOf(0.0))

  /**
   * The scalar parameter used to normalize the merged vectors.
   */
  val normScalar: Double get() = this.normScalarParam.values[0]

  /**
   * Serialize this [BERTParameters] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BERTParameters]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
