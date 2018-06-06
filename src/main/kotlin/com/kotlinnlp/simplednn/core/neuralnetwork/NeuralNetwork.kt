/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The NeuralNetwork.
 *
 * The topology of the Neural Network refers to the way the Neurons are connected each other.
 * This information is contained in the [layersConfiguration].
 *
 * @property layersConfiguration a list of configurations, one per layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class NeuralNetwork(
  val layersConfiguration: List<LayerInterface>,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  /**
   * Secondary constructor.
   *
   * @param layerConfiguration the layersConfiguration of a layer
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   *
   * @return a new NeuralNetwork
   */
  constructor(
    vararg layerConfiguration: LayerInterface,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer()
  ): this(
    layersConfiguration = layerConfiguration.toList(),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [NeuralNetwork] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [NeuralNetwork]
     *
     * @return the [NeuralNetwork] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): NeuralNetwork = Serializer.deserialize(inputStream)
  }

  /**
   * The type of the input array.
   */
  val inputType: LayerType.Input = this.layersConfiguration.first().type

  /**
   * Whether the input array is sparse binary.
   */
  val sparseInput: Boolean = this.inputType == LayerType.Input.SparseBinary

  /**
   * The size of the input, meaningful when the first layer is not a Merge layer.
   */
  val inputSize: Int = this.layersConfiguration.first().size

  /**
   * The size of the inputs, meaningful when the first layer is a Merge layer.
   */
  val inputsSize: List<Int> = this.layersConfiguration.first().sizes

  /**
   * The output size.
   */
  val outputSize: Int = this.layersConfiguration.last().size

  /**
   * The model containing all the trainable parameters of the network.
   */
  val model: NetworkParameters = this.parametersFactory(
    forceDense = true,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * Serialize this [NeuralNetwork] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [NeuralNetwork]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * Generate [NetworkParameters] compatible with the configuration of this network
   *
   * @param forceDense force all parameters to be dense (false by default)
   * @param weightsInitializer the initializer of the weights (null by default)
   * @param biasesInitializer the initializer of the biases (null by default)
   *
   * @return an object containing parameters compatible with the configuration of this network
   */
  fun parametersFactory(forceDense: Boolean,
                        weightsInitializer: Initializer? = null,
                        biasesInitializer: Initializer? = null) = NetworkParameters(
    layersConfiguration = this.layersConfiguration,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer,
    forceDense = forceDense
  )
}
