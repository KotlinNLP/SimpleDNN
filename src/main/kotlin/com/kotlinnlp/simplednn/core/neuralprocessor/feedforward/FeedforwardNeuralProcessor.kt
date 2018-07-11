/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.feedforward.FeedforwardNetworkStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The NeuralProcessor that acts on a Feed-forward Neural Network.
 *
 * @property neuralNetwork a [NeuralNetwork]
 * @property useDropout whether to apply the dropout during the [forward]
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @param mePropK a list of k factors (one per layer) of the 'meProp' algorithm to propagate from the k (in
 *                percentage) output nodes with the top errors of each layer (the list and each element can be null)
 * @property id an identification number useful to track a specific processor
 */
class FeedforwardNeuralProcessor<InputNDArrayType : NDArray<InputNDArrayType>>(
  val neuralNetwork: NeuralNetwork,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  private val mePropK: List<Double?>? = null,
  override val id: Int = 0
) :  NeuralProcessor<
  InputNDArrayType, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  DenseNDArray, // InputErrorsType
  NetworkParameters // ParamsType
  > {

  /**
   * The structure as support of forward and backward.
   */
  var structure = FeedforwardNetworkStructure<InputNDArrayType>(
    layersConfiguration = this.neuralNetwork.layersConfiguration,
    params = this.neuralNetwork.model)

  /**
   * The structure in which to save the contributions of the calculations during the forward (needed to calculate the
   * relevance of the input in respect of the output)
   */
  private val forwardContributions: NetworkParameters by lazy {
    this.neuralNetwork.parametersFactory(forceDense = false)
  }

  /**
   * The errors of the network model parameters
   */
  private val backwardParamsErrors: NetworkParameters by lazy {
    this.neuralNetwork.parametersFactory(forceDense = false)
  }

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
  override fun getParamsErrors(copy: Boolean): NetworkParameters {

    val paramsError: NetworkParameters

    if (copy) {
      paramsError = this.neuralNetwork.parametersFactory(forceDense = false)
      paramsError.assignValues(this.backwardParamsErrors)

    } else {
      paramsError = this.backwardParamsErrors
    }

    return paramsError
  }

  /**
   * Get the errors of the input.
   * This method must be used when the input layer is not a Merge layer.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  override fun getInputErrors(copy: Boolean): DenseNDArray {
    require(!this.neuralNetwork.sparseInput) { "Input errors available only if input is dense" }
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
    require(!this.neuralNetwork.sparseInput) { "Input errors available only if input is dense" }
    require(this.structure.inputLayer is MergeLayer<InputNDArrayType>)

    return (this.structure.inputLayer as MergeLayer<InputNDArrayType>).let {
      if (copy)
        it.inputArrays.map { it.errors.copy() }
      else
        it.inputArrays.map { it.errors }
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

    return this.structure.outputLayer.outputArray.values
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
        networkContributions = this.forwardContributions,
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
        networkContributions = this.forwardContributions,
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
      networkContributions = this.forwardContributions,
      relevantOutcomesDistribution = relevantOutcomesDistribution)

    return this.structure.inputLayer.inputArray.relevance.let { if (copy) it.copy() else it }
  }

  /**
   * The Backward.
   *
   * @param outputErrors the errors to propagate
   */
  override fun backward(outputErrors: DenseNDArray) =
    this.structure.backward(
      outputErrors = outputErrors,
      paramsErrors = this.backwardParamsErrors,
      propagateToInput = this.propagateToInput,
      mePropK = this.mePropK)

  /**
   * Backward errors saving the parameters errors into a given object.
   *
   * @param outputErrors the errors of the output
   * @param paramsErrors the object in which to save the parameters errors
   */
  fun backward(outputErrors: DenseNDArray, paramsErrors: NetworkParameters) =
    this.structure.backward(
      outputErrors = outputErrors,
      paramsErrors = paramsErrors,
      propagateToInput = this.propagateToInput,
      mePropK = this.mePropK)
}
