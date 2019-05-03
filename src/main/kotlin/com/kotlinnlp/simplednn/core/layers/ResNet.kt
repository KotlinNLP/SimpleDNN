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
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
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
 */
class ResNet<InputNDArrayType : NDArray<InputNDArrayType>>(
    val layersConfiguration: List<LayerInterface>,
    val paramsPerLayer: List<LayerParameters<*>>,
    val sumFeedForwardParams: FeedforwardLayerParameters,
    val outputActivation: ActivationFunction? = null
) : LayerContextWindow {

  /**
   * The feedforward layer to reduce input dimension, if the stacked layers output is different.
   */
  private val sumLayer: FeedforwardLayer<InputNDArrayType> = FeedforwardLayer(
      inputArray = AugmentedArray(this.layersConfiguration.first().size),
      outputArray = AugmentedArray.zeros(this.layersConfiguration.last().size),
      params = this.sumFeedForwardParams,
      inputType = LayerType.Input.Dense)

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

    return if (this.layersConfiguration.last().size != this.layersConfiguration.first().size){

      this.sumLayer.setInput(input)
      this.sumLayer.forward()
      this.outputLayer.outputArray.valuesNotActivated.assignSum(this.sumLayer.outputArray.values)
      this.outputActivation!!.f(this.outputLayer.outputArray.valuesNotActivated)

    }else{

      this.outputLayer.outputArray.valuesNotActivated.assignSum(this.inputLayer.inputArray.values as DenseNDArray)
      this.outputActivation!!.f(this.outputLayer.outputArray.valuesNotActivated)

    }
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
   * Propagate the output error using the gradient descent algorithm
   *
   * @param outputErrors the errors to propagate from the output
   * @param propagateToInput whether to propagate the errors to the input
   *
   * @return the params errors
   */
  fun backward(outputErrors: DenseNDArray,
               propagateToInput: Boolean = false): ParamsErrorsList {

    val outErrors: DenseNDArray
    if (outputActivation == null) {
      this.outputLayer.setErrors(outputErrors)
    } else {
      outErrors = outputErrors.assignProd(this.outputActivation.df(outputErrors))
      this.outputLayer.setErrors(outErrors)
      this.sumLayer.setErrors(outErrors)
    }

    val paramsErrorsPerLayer = mutableListOf<List<ParamsArray.Errors<*>>>()

    if (this.layersConfiguration.last().size != this.layersConfiguration.first().size) {
      paramsErrorsPerLayer.add(this.sumLayer.backward(propagateToInput = true))
    }

    for ((i, layer) in this.layers.withIndex().reversed()) {

      this.curLayerIndex = i

      paramsErrorsPerLayer.add(
          layer.backward(propagateToInput = (i > 0 || propagateToInput)))

      if (i == 0)
        if (outputActivation == null)
          layer.inputArray.errors.assignSum(outputErrors)
        else
          layer.inputArray.errors.assignSum(this.sumLayer.inputArray.errors)

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
   * Set the given params errors collector [c] to all [layers].
   *
   * @param c a collector of params errors
   */
  fun setParamsErrorsCollector(c: ParamsErrorsCollector) { this.layers.forEach { it.setParamsErrorsCollector(c) } }

  /**
   * @return the list of layers generated from the [layersConfiguration]
   */
  private fun buildLayers(): List<Layer<*>> = this.layersConfiguration.toLayers(
      paramsPerLayer = this.paramsPerLayer,
      contextWindow = this
  )
}
