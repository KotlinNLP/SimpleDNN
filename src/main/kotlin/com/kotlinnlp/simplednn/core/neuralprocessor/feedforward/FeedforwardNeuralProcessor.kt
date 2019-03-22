/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.StackedLayers
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The NeuralProcessor that acts on stacked-layers networks.
 *
 * @property model the stacked-layers parameters
 * @property useDropout whether to apply the dropout during the [forward]
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property id an identification number useful to track a specific processor
 */
class FeedforwardNeuralProcessor<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: StackedLayersParameters,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) :  NeuralProcessor<
  InputNDArrayType, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  DenseNDArray // InputErrorsType
  > {

  /**
   * The structure as support of forward and backward.
   */
  var structure = StackedLayers<InputNDArrayType>(
    layersConfiguration = this.model.layersConfiguration,
    paramsPerLayer = this.model.paramsPerLayer)

  /**
   * The structure in which to save the contributions of the calculations during the forward (needed to calculate the
   * relevance of the input in respect of the output)
   */
  private val forwardContributions: StackedLayersParameters by lazy {
    StackedLayersParameters(
      layersConfiguration = this.model.layersConfiguration,
      weightsInitializer = null,
      biasesInitializer = null,
      forceDense = false)
  }

  /**
   * The errors of the network model parameters
   */
  private var backwardParamsErrors: ParamsErrorsList = listOf()

  /**
   * @param copy a Boolean indicating whether the returned array must be a copy or a reference
   *
   * @return the output array of the network
   */
  fun getOutput(copy: Boolean = true): DenseNDArray =
    if (copy)
      this.structure.outputLayer.outputArray.values.copy()
    else
      this.structure.outputLayer.outputArray.values

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the network parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    if (copy)
      this.backwardParamsErrors.map { it.copy() }
    else
      this.backwardParamsErrors


  /**
   * Get the errors of the input.
   * This method must be used when the input layer is not a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  override fun getInputErrors(copy: Boolean): DenseNDArray {
    require(!this.model.sparseInput) { "Input errors available only if input is dense" }
    require(this.structure.inputLayer !is MergeLayer<InputNDArrayType>)

    return this.structure.inputLayer.inputArray.errors.let { if (copy) it.copy() else it }
  }

  /**
   * Get the errors of the inputs.
   * This method must be used when the input layer is a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the list of errors of the inputs
   */
  fun getInputsErrors(copy: Boolean = true): List<DenseNDArray> {
    require(!this.model.sparseInput) { "Input errors available only if input is dense" }
    require(this.structure.inputLayer is MergeLayer<InputNDArrayType>)

    return (this.structure.inputLayer as MergeLayer<InputNDArrayType>).let { layer ->
      if (copy)
        layer.inputArrays.map { it.errors.copy() }
      else
        layer.inputArrays.map { it.errors }
    }
  }

  /**
   * The Forward.
   *
   * @param input the input features to forward from the input to the output
   *
   * @return the output array
   */
  override fun forward(input: InputNDArrayType): DenseNDArray {

    this.structure.forward(input = input, useDropout = this.useDropout)

    return this.structure.outputLayer.outputArray.values // TODO: check copy
  }

  /**
   * Forward features, saving the contributes of the input in respect of the output.
   *
   * @param features the features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   *
   * @return the output array
   */
  fun forward(features: InputNDArrayType,
              saveContributions: Boolean): DenseNDArray {

    if (saveContributions)
      this.structure.forward(
        input = features,
        stackedLayersContributions = this.forwardContributions,
        useDropout = this.useDropout)
    else
      this.structure.forward(
        input = features,
        useDropout = this.useDropout)

    return this.structure.outputLayer.outputArray.values
  }

  /**
   * Forward features when the input layer is a Merge layer.
   *
   * @param featuresList the list of features to forward from the input to the output
   *
   * @return the output array
   */
  fun forward(featuresList: List<InputNDArrayType>): DenseNDArray {

    this.structure.forward(input = featuresList, useDropout = this.useDropout)

    return this.structure.outputLayer.outputArray.values
  }

  /**
   * Forward features when the input layer is a Merge layer, saving the contributions of the input to the output.
   *
   * @param featuresList the list of features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to the output (needed to calculate the
   *                          relevance)
   *
   * @return the output array
   */
  fun forward(featuresList: List<InputNDArrayType>,
              saveContributions: Boolean): DenseNDArray {

    if (saveContributions)
      this.structure.forward(
        input = featuresList,
        stackedLayersContributions = this.forwardContributions,
        useDropout = this.useDropout)
    else
      this.structure.forward(
        input = featuresList,
        useDropout = this.useDropout)

    return this.structure.outputLayer.outputArray.values
  }

  /**
   * Calculate the relevance of the input respect to the output, propagating backward the given distribution on the
   * outcomes.
   *
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   * @param copy whether to return a copy of the relevance or not (default = true)
   *
   * @return the input relevance array (If the input is Dense it is Dense, if the input is Sparse or SparseBinary it
   *         is Sparse)
   */
  fun calculateInputRelevance(relevantOutcomesDistribution: DistributionArray, copy: Boolean = true): NDArray<*> {

    this.structure.propagateRelevance(
      stackedLayersContributions = this.forwardContributions,
      relevantOutcomesDistribution = relevantOutcomesDistribution)

    return this.structure.inputLayer.inputArray.relevance.let { if (copy) it.copy() else it }
  }

  /**
   * The Backward.
   *
   * @param outputErrors the errors to propagate
   */
  override fun backward(outputErrors: DenseNDArray) {

    this.backwardParamsErrors = this.structure.backward(
      outputErrors = outputErrors,
      propagateToInput = this.propagateToInput)
  }
}
