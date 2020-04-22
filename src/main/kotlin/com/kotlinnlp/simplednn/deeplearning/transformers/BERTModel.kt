/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The BERT model.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @property attentionOutputSize the size of the attention outputs
 * @property outputHiddenSize the number of the hidden nodes of the output feed-forward
 * @property multiHeadStack the number of scaled-dot attention layers
 * @param dropout the probability of attention dropout (default 0.0)
 * @param numOfLayers the number of stacked layers
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTModel(
  val inputSize: Int,
  val attentionSize: Int,
  val attentionOutputSize: Int,
  val outputHiddenSize: Int,
  val multiHeadStack: Int,
  dropout: Double = 0.0,
  numOfLayers: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read [BERTModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [BERTModel]
     *
     * @return the [BERTModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): BERTModel = Serializer.deserialize(inputStream)

    /**
     * Build a classifier for the classification of masked tokens.
     * It can be used for training and test purpose.
     *
     * @param params the classifier parameters
     *
     * @return a classifier of masked tokens
     */
    fun buildClassifier(params: FeedforwardLayerParameters) = FeedforwardLayer(
      inputArray = AugmentedArray.zeros(size = params.inputSize),
      outputArray = AugmentedArray.zeros(size = params.outputSize),
      params = params,
      inputType = LayerType.Input.Dense,
      activationFunction = Softmax())
  }

  /**
   * The parameters of the stacked layers.
   */
  val layers: List<BERTParameters> = List(
    size = numOfLayers,
    init = {
      BERTParameters(
        inputSize = this.inputSize,
        attentionSize = this.attentionSize,
        attentionOutputSize = this.attentionOutputSize,
        outputHiddenSize = this.outputHiddenSize,
        multiHeadStack = this.multiHeadStack,
        dropout = dropout,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer)
    }
  )

  /**
   * The embeddings used to train the model, useful to save them in the same structure.
   */
  var embeddingsMap: EmbeddingsMap<String>? = null

  /**
   * Serialize this [BERTModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BERTModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
