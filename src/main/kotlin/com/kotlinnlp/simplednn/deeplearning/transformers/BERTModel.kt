/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.GeLU
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.utils.DictionarySet
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
 * @property numOfHeads the number of self-attention heads
 * @property vocabulary the vocabulary with all the well-known forms of the model (forms not present in it are treated
 *                      as unknown)
 * @param wordEmbeddings pre-trained word embeddings or null to generate them randomly using the [vocabulary]
 * @param numOfLayers the number of stacked layers
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class BERTModel(
  val inputSize: Int,
  val attentionSize: Int,
  val attentionOutputSize: Int,
  val outputHiddenSize: Int,
  val numOfHeads: Int,
  val vocabulary: DictionarySet<String>,
  wordEmbeddings: EmbeddingsMap<String>? = null,
  numOfLayers: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  /**
   * Functional token.
   */
  enum class FuncToken(val form: String) {
    CLS("[CLS]"),
    SEP("[SEP]"),
    PAD("[PAD]"),
    UNK("[UNK]"),
    MASK("[MASK]");

    companion object {

      /**
       * The [FuncToken] associated by form.
       */
      private val tokensByForm: Map<String, FuncToken> = values().associateBy { it.form }

      /**
       * @param form a token form
       *
       * @return the [FuncToken] with the given form
       */
      fun byForm(form: String) = tokensByForm.getValue(form)
    }
  }

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
        numOfHeads = this.numOfHeads,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer)
    }
  )

  /**
   * The parameters of the embeddings norm layer.
   */
  val embNorm = StackedLayersParameters(
    LayerInterface(size = this.inputSize),
    LayerInterface(size = this.inputSize, connectionType = LayerType.Connection.Norm),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The word embeddings.
   * If not trained, they can be set to null before a model serialization and re-set after deserialization, in order to
   * make the model lighter.
   */
  var wordEmb: EmbeddingsMap<String>? = wordEmbeddings ?: EmbeddingsMap<String>(this.inputSize).apply {
    vocabulary.getElements().forEach { set(it) }
  }

  /**
   * The functional embeddings associated to the [FuncToken].
   */
  var funcEmb: EmbeddingsMap<FuncToken> = EmbeddingsMap<FuncToken>(this.inputSize).apply {
    FuncToken.values().forEach {
      set(key = it, embedding = wordEmb!!.getOrNull(it.form))
    }
  }

  /**
   * The positional embeddings.
   */
  val positionalEmb: EmbeddingsMap<Int> = EmbeddingsMap(this.inputSize)

  /**
   * The token type embeddings.
   */
  val tokenTypeEmb: EmbeddingsMap<Int> = EmbeddingsMap<Int>(this.inputSize).apply {
    set(0)
    set(1)
  }

  /**
   * The model of the classifier used to train the model.
   */
  var classifier: StackedLayersParameters = StackedLayersParameters(
    LayerInterface(size = inputSize),
    LayerInterface(size = inputSize, connectionType = LayerType.Connection.Feedforward, activationFunction = GeLU),
    LayerInterface(size = inputSize, connectionType = LayerType.Connection.Norm),
    LayerInterface(
      size = vocabulary.size, connectionType = LayerType.Connection.Feedforward, activationFunction = Softmax())
  )

  /**
   * Serialize this [BERTModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BERTModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
