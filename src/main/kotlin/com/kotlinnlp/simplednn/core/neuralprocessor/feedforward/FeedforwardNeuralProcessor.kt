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
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The neural processor that acts on networks of stacked-layers.
 *
 * @property model the stacked-layers parameters
 * @param dropouts the probability of dropout for each stacked layer
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @param paramsErrorsCollector where to collect the local params errors during the [backward] (optional)
 * @property id an identification number useful to track a specific processor
 */
class FeedforwardNeuralProcessor<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: StackedLayersParameters,
  dropouts: List<Double>,
  override val propagateToInput: Boolean,
  private val paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector(),
  override val id: Int = 0
) : NeuralProcessor<
  InputNDArrayType, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  DenseNDArray // InputErrorsType
  > {

  /**
   * The neural processor that acts on networks of stacked-layers.
   *
   * @param model the stacked-layers parameters
   * @param dropout the probability of dropout for each stacked layer (default 0.0)
   * @param propagateToInput whether to propagate the errors to the input during the [backward]
   * @param paramsErrorsCollector where to collect the local params errors during the [backward] (optional)
   * @param id an identification number useful to track a specific processor
   */
  constructor(
    model: StackedLayersParameters,
    dropout: Double = 0.0,
    propagateToInput: Boolean,
    paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector(),
    id: Int = 0
  ): this(
    model = model,
    dropouts = List(model.numOfLayers) { dropout },
    propagateToInput = propagateToInput,
    paramsErrorsCollector = paramsErrorsCollector,
    id = id
  )

  /**
   * The stacked layers.
   */
  private var layers = StackedLayers<InputNDArrayType>(params = this.model, dropouts = dropouts).apply {
    setParamsErrorsCollector(paramsErrorsCollector)
  }

  /**
   * The support in which to save the contributions of the input to the output during a forward (used to calculate the
   * relevance).
   */
  private val contributions: StackedLayersParameters by lazy {
    StackedLayersParameters(
      layersConfiguration = this.model.layersConfiguration,
      weightsInitializer = null,
      biasesInitializer = null)
  }

  /**
   * The model parameters errors calculated with a [backward].
   */
  private var paramsErrors: ParamsErrorsList = listOf()

  /**
   * @param copy a Boolean indicating whether the returned array must be a copy or a reference
   *
   * @return the output array of the network
   */
  fun getOutput(copy: Boolean = true): DenseNDArray =
    if (copy)
      this.layers.outputLayer.outputArray.values.copy()
    else
      this.layers.outputLayer.outputArray.values

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the network parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    if (copy)
      this.paramsErrors.map { it.copy() }
    else
      this.paramsErrors


  /**
   * Get the errors of the input.
   * This method must be used only when the input layer is not a [MergeLayer].
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  override fun getInputErrors(copy: Boolean): DenseNDArray {
    require(!this.model.sparseInput) { "Input errors available only if input is dense" }
    require(this.layers.inputLayer !is MergeLayer<InputNDArrayType>)

    return this.layers.inputLayer.inputArray.errors.let { if (copy) it.copy() else it }
  }

  /**
   * Get the errors of the inputs.
   * This method must be used only when the input layer is a [MergeLayer].
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the list of errors of the inputs
   */
  fun getInputsErrors(copy: Boolean = true): List<DenseNDArray> {

    require(!this.model.sparseInput) { "The input errors available only if the input is not sparse." }
    require(this.layers.inputLayer is MergeLayer<InputNDArrayType>)

    return (this.layers.inputLayer as MergeLayer<InputNDArrayType>).let { layer ->
      if (copy)
        layer.inputArrays.map { it.errors.copy() }
      else
        layer.inputArrays.map { it.errors }
    }
  }

  /**
   * Execute the forward of the input to the output.
   *
   * @param input the input
   *
   * @return the output
   */
  override fun forward(input: InputNDArrayType): DenseNDArray {

    this.layers.forward(input)

    return this.layers.outputLayer.outputArray.values // TODO: check copy
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
  fun forward(features: InputNDArrayType, saveContributions: Boolean): DenseNDArray {

    if (saveContributions)
      this.layers.forward(features, contributions = this.contributions)
    else
      this.layers.forward(features)

    return this.layers.outputLayer.outputArray.values
  }

  /**
   * Forward features when the input layer is a [MergeLayer].
   *
   * @param featuresList the list of features to forward from the input to the output
   *
   * @return the output array
   */
  fun forward(featuresList: List<InputNDArrayType>): DenseNDArray {

    this.layers.forward(featuresList)

    return this.layers.outputLayer.outputArray.values
  }

  /**
   * Forward features when the input layer is a [MergeLayer], saving the contributions of the input to the output.
   *
   * @param featuresList the list of features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to the output (needed to calculate the
   *                          relevance)
   *
   * @return the output array
   */
  fun forward(featuresList: List<InputNDArrayType>, saveContributions: Boolean): DenseNDArray {

    if (saveContributions)
      this.layers.forward(featuresList, contributions = this.contributions)
    else
      this.layers.forward(featuresList)

    return this.layers.outputLayer.outputArray.values
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

    this.layers.propagateRelevance(
      layersContributions = this.contributions,
      relevantOutcomesDistribution = relevantOutcomesDistribution)

    return this.layers.inputLayer.inputArray.relevance.let { if (copy) it.copy() else it }
  }

  /**
   * Propagate the output errors with a stochastic gradient descent (SGD) algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: DenseNDArray) {

    this.paramsErrors = this.layers.backward(
      outputErrors = outputErrors,
      propagateToInput = this.propagateToInput)
  }
}
