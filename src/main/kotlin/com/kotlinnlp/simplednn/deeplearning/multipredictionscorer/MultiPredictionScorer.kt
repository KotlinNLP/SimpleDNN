/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The MultiPredictionScorer.
 *
 * @property model the neural model of this scorer
 */
class MultiPredictionScorer<InputNDArrayType : NDArray<InputNDArrayType>>(val model: MultiPredictionModel) {

  /**
   * The list of processors pools, one for each sub-network.
   */
  private val processorsPools = List<FeedforwardNeuralProcessorsPool<InputNDArrayType>>(
    size = this.model.networks.size,
    init = { i -> FeedforwardNeuralProcessorsPool(this.model.networks[i]) }
  )

  /**
   * A multimap of processors used during the last scoring.
   */
  private lateinit var usedProcessors: MultiMap<FeedforwardNeuralProcessor<InputNDArrayType>>

  /**
   * A multimap of processors for which a backward was called the last time.
   */
  private lateinit var backwardedProcessors: MultiMap<FeedforwardNeuralProcessor<InputNDArrayType>>

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a multimap of params errors, one for each 'backwarded' processor
   */
  fun getParamsErrors(copy: Boolean): MultiMap<NetworkParameters> {

    return this.backwardedProcessors.map { _, _, processor ->
      processor.getParamsErrors(copy = copy)
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a multimap of input errors, one for each 'backwarded' processor
   */
  fun getInputErrors(copy: Boolean = true): MultiMap<DenseNDArray> {

    return this.backwardedProcessors.map { _, _, processor ->
      processor.getInputErrors(copy = copy)
    }
  }

  /**
   * Get the output score arrays given the inputs.
   *
   * @param featuresMap a multimap of input features, one for each prediction
   * @param useDropout whether to apply the dropout
   *
   * @return a multimap of output arrays, one for each prediction done
   */
  fun score(featuresMap: MultiMap<InputNDArrayType>, useDropout: Boolean = false): MultiMap<DenseNDArray> {

    this.checkInputMapKeys(featuresMap)

    this.initUsedProcessors(featuresMap)

    return featuresMap.map { i, j, features ->
      this.usedProcessors[i, j]!!.forward(features, useDropout = useDropout)
    }
  }


  /**
   * Get the output score arrays given the inputs, saving contributions to calculate the input relevance.
   *
   * @param featuresMap a multimap of input features, one for each prediction
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param useDropout whether to apply the dropout
   *
   * @return a multimap of output arrays
   */
  fun score(featuresMap: MultiMap<InputNDArrayType>,
            saveContributions: Boolean,
            useDropout: Boolean = false): MultiMap<DenseNDArray> {

    this.checkInputMapKeys(featuresMap)

    this.initUsedProcessors(featuresMap)

    return featuresMap.map { i, j, features ->
      this.usedProcessors[i, j]!!.forward(features, saveContributions= saveContributions, useDropout = useDropout)
    }
  }

  /**
   * Calculate the relevance of the input respect to the output, propagating backward the given distribution on the
   * outcomes.
   *
   * @param relevantOutcomesDistribution a multimap of output relevance distributions, one for each prediction, which
   *                                     indicate which outcomes are relevant, used as reference to calculate the
   *                                     relevance of the input
   * @param copy whether to return a copy of the relevance or not
   *
   * @return a multimap of input relevance arrays (if the input is Dense they are Dense, if the input is Sparse or
   *         SparseBinary they are Sparse)
   */
  fun calculateInputRelevance(relevantOutcomesDistribution: MultiMap<DistributionArray>,
                              copy: Boolean = true): MultiMap<NDArray<*>> {

    this.checkInputMapKeys(relevantOutcomesDistribution)
    this.checkInputMapUsedKeys(relevantOutcomesDistribution)
    this.checkInputMapPredictionIndices(relevantOutcomesDistribution)

    return relevantOutcomesDistribution.map { i, j, outcomesDistribution ->
      this.usedProcessors[i, j]!!.calculateInputRelevance(outcomesDistribution, copy = copy)
    }
  }

  /**
   * Backward the errors of the given predictions.
   *
   * @param outputsErrors a multimap of outputs errors
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputsErrors: MultiMap<DenseNDArray>, propagateToInput: Boolean) {

    this.checkInputMapKeys(outputsErrors)
    this.checkInputMapUsedKeys(outputsErrors)
    this.checkInputMapPredictionIndices(outputsErrors)

    this.setBackwardedProcessors(outputsErrors)

    outputsErrors.forEach { i, j, errors ->
      this.usedProcessors[i, j]!!.backward(errors, propagateToInput = propagateToInput)
    }
  }

  /**
   * Check if the input multimap keys are compatible with the model.
   *
   * @param inputMultiMap a multimap with sub-networks indices as keys
   *
   * @throws IllegalArgumentException if the indices of the map are negative or exceed the last sub-network index
   */
  private fun checkInputMapKeys(inputMultiMap: MultiMap<*>) {

    inputMultiMap.keys.forEach { networkIndex ->
      require(networkIndex in 0 until this.model.networks.size) {
        "Index %d not in range [0, %d]".format(networkIndex, this.model.networks.lastIndex)
      }
    }
  }

  /**
   * Check if the input multimap keys are compatible with the networks used for the last scoring.
   *
   * @param inputMultiMap a multimap with sub-networks indices as keys
   *
   * @throws IllegalArgumentException if the indices of the map are not within the ones used for the last scoring
   */
  private fun checkInputMapUsedKeys(inputMultiMap: MultiMap<*>) {

    val networkIndices: Set<Int> = this.usedProcessors.keys

    inputMultiMap.keys.forEach { networkIndex ->
      require(networkIndex in networkIndices) { "Network $networkIndex not used" }
    }
  }

  /**
   * Check if the prediction indices of an input multimap are compatible with the last predictions done.
   *
   * @param inputMultiMap a multimap of sub-networks indices to maps of prediction indices to objects
   *
   * @throws IllegalArgumentException if a prediction index is not compatible with the last predictions done
   */
  private fun checkInputMapPredictionIndices(inputMultiMap: MultiMap<*>) {

    inputMultiMap.forEach { networkIndex, predictionIndex, _ ->

        val processorsMap: Map<Int, FeedforwardNeuralProcessor<InputNDArrayType>> = this.usedProcessors[networkIndex]!!

        require(predictionIndex in 0 until processorsMap.size ) {
          "%d predictions done with the network %d, but %d given as prediction index"
            .format(networkIndex, processorsMap.size, predictionIndex)
        }
    }
  }

  /**
   * Initialize used processors.
   *
   * @param featuresMap the input features multimap
   */
  private fun initUsedProcessors(featuresMap: MultiMap<InputNDArrayType>) {

    this.processorsPools.forEach { it.releaseAll() }

    this.usedProcessors = featuresMap.map { networkIndex, _, _ -> this.processorsPools[networkIndex].getItem() }
  }

  /**
   * Set the processors for which a backward was called last time.
   *
   * @param outputErrors the output errors multimap
   */
  private fun setBackwardedProcessors(outputErrors: MultiMap<DenseNDArray>) {

    this.backwardedProcessors = outputErrors.map { i, j, _ -> this.usedProcessors[i, j]!! }
  }
}
