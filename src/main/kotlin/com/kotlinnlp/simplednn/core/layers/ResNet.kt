/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * Residual Network cass.
 * For details see: Study of Residual Networks for Image Recognition
 *
 * @property layersConfiguration the layers configurations
 * @property paramsPerLayer the parameters per layer
 * @property sumFeedForwardParams a feed forward layer, the purpose of which is reduce the input to the output size.
 * @property outputActivation the activation function on resNet output (after the sum)
 */
class ResNet<InputNDArrayType : NDArray<InputNDArrayType>>(
    val layersConfiguration: List<LayerInterface>,
    val paramsPerLayer: List<LayerParameters<*>>,
    val sumFeedForwardParams: FeedforwardLayerParameters?,
    val outputActivation: ActivationFunction
) : LayerContextWindow {

  /**
   * The feed forward layer to reduce input dimension, if the stacked layers output is different.
   */
  private val sumLayer: FeedforwardLayer<InputNDArrayType>? =
      if (sumFeedForwardParams != null)
        FeedforwardLayer(
            inputArray = AugmentedArray(this.layersConfiguration.first().size),
            outputArray = AugmentedArray.zeros(this.layersConfiguration.last().size),
            params = this.sumFeedForwardParams,
            inputType = LayerType.Input.Dense)
      else
        null

  /**
   * The list of layers generate from the [layersConfiguration].
   */
  val layers: List<Layer<*>> = this.buildLayers()

  /**
   * Used to track the loop over the layers during the forward and the backward
   */
  var curLayerIndex: Int = 0

  /**
   * The first layer
   */
  @Suppress("UNCHECKED_CAST")
  val inputLayer: Layer<InputNDArrayType> get() = layers.first() as Layer<InputNDArrayType>

  /**
   * The last layer
   */
  @Suppress("UNCHECKED_CAST")
  val outputLayer: Layer<DenseNDArray> get() = layers.last() as Layer<DenseNDArray>

  /**
   * Perform the sum operation on Output.
   *
   * @param input the input to forward from the input to the output
   *
   * @return the output [NDArray]
   */
  private fun getOutput(input: InputNDArrayType): DenseNDArray {

    if (this.layersConfiguration.last().size != this.layersConfiguration.first().size){

      this.sumLayer!!.setInput(input)
      this.sumLayer.forward()
      this.outputLayer.outputArray.valuesNotActivated.assignSum(this.sumLayer.outputArray.values)

    }else{

      this.outputLayer.outputArray.valuesNotActivated.assignSum(this.inputLayer.inputArray.values as DenseNDArray)
    }

    return this.outputActivation.f(this.outputLayer.outputArray.valuesNotActivated)
  }

  /**
   * Forward features.
   *
   * @param input the input to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the output [NDArray]
   */
  fun forward(input: InputNDArrayType, useDropout: Boolean = false): DenseNDArray {

    this.inputLayer.setInput(input)

    for ((i, layer) in this.layers.withIndex()) {
      this.curLayerIndex = i
      layer.forward(useDropout = useDropout)
    }

    return this.getOutput(input)

  }

  /**
   * Propagate the output error using the gradient descent algorithm on stacked layers and on input,
   * trough [sumLayer].
   *
   * @param outputErrors the errors to propagate from the output
   *
   */
  private fun setOutputError(outputErrors: DenseNDArray) {

    if (this.layersConfiguration.last().size != this.layersConfiguration.first().size) {

      this.outputLayer.setErrors(outputErrors)
      this.sumLayer!!.setErrors(outputErrors)

    } else {

      this.outputLayer.setErrors(outputErrors)
    }

  }

  /**
   * Propagate the output error using the gradient descent algorithm
   *
   * @param outputErrors the errors to propagate from the output
   * @param propagateToInput whether to propagate the errors to the input
   *
   * @return the params errors
   */
  fun backward(outputErrors: DenseNDArray,
               propagateToInput: Boolean = false): ParamsErrorsList {

    val outErrorsActDerivative: DenseNDArray = this.outputActivation.df(outputErrors)

    this.setOutputError(outputErrors.assignProd(outErrorsActDerivative))

    val paramsErrorsPerLayer = mutableListOf<List<ParamsArray.Errors<*>>>()

    if (this.layersConfiguration.last().size != this.layersConfiguration.first().size) {
      paramsErrorsPerLayer.add(this.sumLayer!!.backward(propagateToInput = true))
    }

    for ((i, layer) in this.layers.withIndex().reversed()) {

      this.curLayerIndex = i

      paramsErrorsPerLayer.add(
          layer.backward(propagateToInput = (i > 0 || propagateToInput)))

      if (i == 0)
        if (this.layersConfiguration.last().size != this.layersConfiguration.first().size)
          layer.inputArray.errors.assignSum(this.sumLayer!!.inputArray.errors)
        else
          layer.inputArray.errors.assignSum(outputErrors)

    }

    return paramsErrorsPerLayer.flatten()
  }

  /**
   * @return the current layer in previous state
   */
  override fun getPrevState(): Layer<*>? = null

  /**
   * @return the current layer in next state
   */
  override fun getNextState(): Layer<*>? = null

  /**
   * @return the list of layers generated from the [layersConfiguration]
   */
  private fun buildLayers(): List<Layer<*>> = this.layersConfiguration.toLayers(
      paramsPerLayer = this.paramsPerLayer,
      contextWindow = this
  )
}
