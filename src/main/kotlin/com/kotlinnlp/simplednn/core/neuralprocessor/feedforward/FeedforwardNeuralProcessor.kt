/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.feedforward.FeedforwardNetworkStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The [FeedforwardNeuralProcessor] acts on the [neuralNetwork] performing predictions
 * and training based on Examples.
 *
 * @property neuralNetwork a [NeuralNetwork]
 * @property id an identification number useful to track a specific processor
 */
class FeedforwardNeuralProcessor<InputNDArrayType : NDArray<InputNDArrayType>>(
  neuralNetwork: NeuralNetwork,
  id: Int = 0
) : NeuralProcessor(neuralNetwork = neuralNetwork, id = id) {

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
  private val forwardContributions: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   * The errors of the network model parameters
   */
  private val backwardParamsErrors: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   * @param copy a Boolean indicating whether the returned array must be a copy or a reference
   *
   * @return the output array of the network
   */
  override fun getOutput(copy: Boolean): DenseNDArray {
    return if (copy) {
      this.structure.outputLayer.outputArray.values.copy()
    } else {
      this.structure.outputLayer.outputArray.values
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the network parameters
   */
  override fun getParamsErrors(copy: Boolean): NetworkParameters {

    val paramsError: NetworkParameters

    if (copy) {
      paramsError = this.neuralNetwork.parametersErrorsFactory()
      paramsError.assignValues(this.backwardParamsErrors)

    } else {
      paramsError = this.backwardParamsErrors
    }

    return paramsError
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  fun getInputErrors(copy: Boolean = true): DenseNDArray {
    require(!this.neuralNetwork.sparseInput) { "Input errors available only if input is dense" }

    return if (copy) {
      this.structure.inputLayer.inputArray.errors.copy()
    } else {
      this.structure.inputLayer.inputArray.errors
    }
  }

  /**
   * Forward features.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the output array
   */
  fun forward(featuresArray: InputNDArrayType, useDropout: Boolean = false): DenseNDArray {

    this.structure.forward(features = featuresArray, useDropout = useDropout)

    return this.structure.outputLayer.outputArray.values
  }

  /**
   * Forward features, saving the contributes of the input in respect of the output.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param useDropout whether to apply the dropout
   *
   * @return the output array
   */
  fun forward(featuresArray: InputNDArrayType,
              saveContributions: Boolean,
              useDropout: Boolean = false): DenseNDArray {

    if (saveContributions) {
      this.structure.forward(
        features = featuresArray,
        networkContributions = this.forwardContributions,
        useDropout = useDropout)

    } else {
      this.structure.forward(features = featuresArray, useDropout = useDropout)
    }

    return this.structure.outputLayer.outputArray.values
  }

  /**
   * Calculate the relevance of the input of respect of the output, propagating backward the given distribution on the
   * outcomes.
   *
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   * @param copy whether to return a copy of the relevance or not
   *
   * @return the input relevance array (If the input is Dense it is Dense, if the input is Sparse or SparseBinary it
   *         is Sparse)
   */
  fun calculateInputRelevance(relevantOutcomesDistribution: DistributionArray, copy: Boolean = true): NDArray<*> {

    this.structure.propagateRelevance(
      networkContributions = this.forwardContributions,
      relevantOutcomesDistribution = relevantOutcomesDistribution)

    return if (copy) {
      this.structure.inputLayer.inputArray.relevance.copy()
    } else {
      this.structure.inputLayer.inputArray.relevance
    }
  }

  /**
   * Backward errors.
   *
   * @param outputErrors the errors of the output
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrors: DenseNDArray, propagateToInput: Boolean = false) {

    this.structure.backward(
      outputErrors = outputErrors,
      paramsErrors = this.backwardParamsErrors,
      propagateToInput = propagateToInput)
  }
}
