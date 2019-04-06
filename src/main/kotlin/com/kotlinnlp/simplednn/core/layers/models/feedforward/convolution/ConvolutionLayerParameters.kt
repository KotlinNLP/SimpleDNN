/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The Convolution Layer Parameters.
 *
 *  @param kernelSize the shape of the parameters (all parameters share this shape)
 *  @param inputChannels the number of inputs
 *  @param outputChannels the number of outputs. The layer has n kernels, where n == outputChannels * inputChannels
 *  @param weightsInitializer the initializer of the weights (kernels)
 *  @param biasesInitializer the initializer of the bias
 */
class ConvolutionLayerParameters(
    val kernelSize: Shape,
    val inputChannels: Int,
    val outputChannels: Int,
    weightsInitializer: Initializer? = GlorotInitializer(),
    biasesInitializer: Initializer? = GlorotInitializer(),
    private val sparseInput: Boolean = false
) : LayerParameters<ConvolutionLayerParameters>(
    inputSize = 0,
    outputSize = 0,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer) {
  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = List(inputChannels * outputChannels)
  {ParamsArray(this.kernelSize.dim1, this.kernelSize.dim2)}

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = List(outputChannels)
  {ParamsArray(1,1)}

  /**
   * The list of all parameters.
   */
  override val paramsList = weightsList + biasesList

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [ConvolutionLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): ConvolutionLayerParameters {

    val clonedParams = ConvolutionLayerParameters(
        kernelSize = this.kernelSize,
        inputChannels = this.inputChannels,
        outputChannels = this.outputChannels,
        sparseInput = this.sparseInput,
        weightsInitializer = null,
        biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }

}
