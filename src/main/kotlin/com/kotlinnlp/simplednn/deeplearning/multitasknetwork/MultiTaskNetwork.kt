/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multitasknetwork

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A multi-task network is composed by single input feed-forward layer shared by more networks, each with a own output
 * feed-forward layer.
 *
 * @property model the model of this network
 */
class MultiTaskNetwork<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: MultiTaskNetworkModel
) {

  /**
   * The neural processor of the input network.
   */
  val inputProcessor = FeedforwardNeuralProcessor<InputNDArrayType>(this.model.inputNetwork)

  /**
   * The list of neural processors of the output networks.
   */
  val outputProcessors: List<FeedforwardNeuralProcessor<DenseNDArray>> =
    this.model.outputNetworks.map { FeedforwardNeuralProcessor<DenseNDArray>(it) }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the neural parameters
   */
  fun getParamsErrors(copy: Boolean) = MultiTaskNetworkParameters(
    inputParams = this.inputProcessor.getParamsErrors(copy = copy),
    outputParamsList = this.outputProcessors.map { it.getParamsErrors(copy = copy) }
  )

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input
   */
  fun getInputErrors(copy: Boolean = true): DenseNDArray = this.inputProcessor.getInputErrors(copy = copy)

  /**
   * Forward features.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param useDropout whether to apply the dropout
   *
   * @return the list of output arrays, one for each output network
   */
  fun forward(featuresArray: InputNDArrayType, useDropout: Boolean = false): List<DenseNDArray> {

    val hiddenOutput: DenseNDArray = this.inputProcessor.forward(featuresArray = featuresArray, useDropout = useDropout)

    return this.outputProcessors.map { it.forward(featuresArray = hiddenOutput, useDropout = useDropout) }
  }

  /**
   * Forward features, saving the contributes of the input in respect of the output.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param useDropout whether to apply the dropout
   *
   * @return the list of output arrays, one for each output network
   */
  fun forward(featuresArray: InputNDArrayType,
              saveContributions: Boolean,
              useDropout: Boolean = false): List<DenseNDArray> {

    val hiddenOutput: DenseNDArray = this.inputProcessor.forward(
      featuresArray = featuresArray,
      saveContributions = saveContributions,
      useDropout = useDropout)

    return this.outputProcessors.map {
      it.forward(featuresArray = hiddenOutput, saveContributions = saveContributions, useDropout = useDropout)
    }
  }

  /**
   * Calculate the relevance of the input respect to the output of the network with the given networkIndex,
   * propagating backward the given distribution on the outcomes.
   *
   * @param networkIndex the index of an output network (starting from 0)
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   * @param copy whether to return a copy of the relevance or not (default = true)
   *
   * @return the input relevance array (If the input is Dense it is Dense, if the input is Sparse or SparseBinary it
   *         is Sparse)
   */
  fun calculateInputRelevance(networkIndex: Int,
                              relevantOutcomesDistribution: DistributionArray,
                              copy: Boolean = true): NDArray<*> {
    require(networkIndex in 0 .. this.outputProcessors.size) {
      "Invalid network index: %d. Must be in range [0, %d].".format(networkIndex, this.outputProcessors.lastIndex)
    }

    val hiddenRelevance: DenseNDArray = this.outputProcessors[networkIndex].calculateInputRelevance(
      relevantOutcomesDistribution = relevantOutcomesDistribution,
      copy = false) as DenseNDArray // the hidden layer is always dense, the same for its relevance

    return this.inputProcessor.calculateInputRelevance(
      relevantOutcomesDistribution = DistributionArray(hiddenRelevance),
      copy = copy)
  }

  /**
   * Backward errors.
   *
   * @param outputErrorsList the list of output errors, one for each output network
   * @param propagateToInput whether to propagate the errors to the input
   * @param inputMePropK the input layer k factor of the 'meProp' algorithm to propagate from the k (in percentage)
   *                     hidden nodes with the top errors (can be null)
   * @param outputMePropK a list of k factors (one for each output layer) of the 'meProp' algorithm to propagate from
   *                      the k (in percentage) output nodes with the top errors (the list and each element can be null)
   */
  fun backward(outputErrorsList: List<DenseNDArray>,
               propagateToInput: Boolean = false,
               inputMePropK: Double? = null,
               outputMePropK: List<Double?>? = null) {

    val hiddenErrors: DenseNDArray =
      this.backwardOutputProcessors(outputErrorsList = outputErrorsList, outputMePropK = outputMePropK)

    this.inputProcessor
      .backward(outputErrors = hiddenErrors, propagateToInput = propagateToInput, mePropK = listOf(inputMePropK))
  }

  /**
   * Output processors backwards.
   *
   * @param outputErrorsList the list of output errors, one for each output network
   * @param outputMePropK a list of k factors (one for each output layer) of the 'meProp' algorithm to propagate from
   *                      the k (in percentage) output nodes with the top errors (the list and each element can be null)
   *
   * @return the sum of the input errors of each output network
   */
  private fun backwardOutputProcessors(outputErrorsList: List<DenseNDArray>,
                                       outputMePropK: List<Double?>? = null): DenseNDArray {

    require(outputErrorsList.size == this.outputProcessors.size) {
      "The list of output errors must have a size equal to the number of output networks."
    }

    var hiddenErrors: DenseNDArray? = null

    this.outputProcessors.zip(outputErrorsList).forEachIndexed { i, (processor, errors) ->

      processor.backward(outputErrors = errors, propagateToInput = true, mePropK = listOf(outputMePropK?.get(i)))

      if (hiddenErrors == null)
        hiddenErrors = processor.getInputErrors(copy = true)
      else
        hiddenErrors!!.assignSum(processor.getInputErrors(copy = false))
    }

    return hiddenErrors!!
  }
}
