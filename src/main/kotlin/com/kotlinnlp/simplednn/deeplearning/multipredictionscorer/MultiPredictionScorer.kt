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
   * A map of sub-networks indices to lists of processors used during the last scoring.
   */
  private val usedProcessorsPerNetwork = mutableMapOf<Int, List<FeedforwardNeuralProcessor<InputNDArrayType>>>()

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a map of sub-networks indices to lists of params errors, one for each prediction done
   */
  fun getParamsErrors(copy: Boolean): Map<Int, List<NetworkParameters>> {
    return this.mapUsedProcessors { _, _, processor -> processor.getParamsErrors(copy = copy) }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return a map of sub-networks indices to lists of input errors, one for each prediction done
   */
  fun getInputErrors(copy: Boolean = true): Map<Int, List<DenseNDArray>> {
    return this.mapUsedProcessors { _, _, processor -> processor.getInputErrors(copy = copy) }
  }

  /**
   * Get the output score arrays given the inputs.
   *
   * @param featuresMap a map of sub-networks indices to lists of input features, one for each prediction
   * @param useDropout whether to apply the dropout
   *
   * @return a map of sub-networks indices to lists of output arrays, one for each prediction done
   */
  fun score(featuresMap: Map<Int, List<InputNDArrayType>>, useDropout: Boolean = false): Map<Int, List<DenseNDArray>> {

    this.checkInputMapKeys(featuresMap)

    this.initProcessors(featuresMap)

    return this.mapUsedProcessors { i, j, processor -> processor.forward(featuresMap[i]!![j], useDropout = useDropout) }
  }


  /**
   * Get the output score arrays given the inputs, saving contributions to calculate the input relevance.
   *
   * @param featuresMap a map of sub-networks indices to lists of input features, one for each prediction
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param useDropout whether to apply the dropout
   *
   * @return a map of sub-networks indices to lists of output arrays
   */
  fun score(featuresMap: Map<Int, List<InputNDArrayType>>,
            saveContributions: Boolean,
            useDropout: Boolean = false): Map<Int, List<DenseNDArray>> {

    this.checkInputMapKeys(featuresMap)

    this.initProcessors(featuresMap)

    return this.mapUsedProcessors { i, j, processor ->
      processor.forward(featuresMap[i]!![j], saveContributions= saveContributions, useDropout = useDropout)
    }
  }

  /**
   * Calculate the relevance of the input respect to the output, propagating backward the given distribution on the
   * outcomes.
   *
   * @param relevantOutcomesDistribution a map of sub-networks indices to lists of output relevance distributions, one
   *                                     for each prediction, which indicate which outcomes are relevant, used as
   *                                     reference to calculate the relevance of the input
   * @param copy whether to return a copy of the relevance or not
   *
   * @return a map of sub-networks indices to maps of prediction indices to input relevance arrays (if the input is
   *         Dense they are Dense, if the input is Sparse or SparseBinary they are Sparse)
   */
  fun calculateInputRelevance(relevantOutcomesDistribution: Map<Int, Map<Int, DistributionArray>>,
                              copy: Boolean = true): Map<Int, Map<Int, NDArray<*>>> {

    this.checkInputMapKeys(relevantOutcomesDistribution)
    this.checkInputMapUsedKeys(relevantOutcomesDistribution)
    this.checkInputMapPredictionIndices(relevantOutcomesDistribution)

    val networkIndices: List<Int> = relevantOutcomesDistribution.keys.toList()

    return mapOf(*Array(
      size = networkIndices.size,
      init = { i ->
        val networkIndex: Int = networkIndices[i]
        val processorsDistributionMap: Map<Int, DistributionArray> = relevantOutcomesDistribution[networkIndex]!!
        val processorsList = this.usedProcessorsPerNetwork[networkIndex]!!

        Pair(
          networkIndex,
          mapOf(*Array(
            size = processorsDistributionMap.size,
            init = { processorsIndex ->

              Pair(
                processorsIndex,
                processorsList[processorsIndex]
                .calculateInputRelevance(processorsDistributionMap[processorsIndex]!!, copy = copy)
              )
            }
          ))
        )
      }
    ))
  }

  /**
   * Backward the errors of the given predictions.
   *
   * @param outputErrors a map of sub-networks indices to maps of prediction indices to outputs errors
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrors: Map<Int, Map<Int, DenseNDArray>>, propagateToInput: Boolean) {

    this.checkInputMapKeys(outputErrors)
    this.checkInputMapUsedKeys(outputErrors)
    this.checkInputMapPredictionIndices(outputErrors)

    outputErrors.forEach { networkIndex, processorsErrors ->
      processorsErrors.forEach { processorIndex, errors ->

        this.usedProcessorsPerNetwork[networkIndex]!![processorIndex]
          .backward(errors, propagateToInput = propagateToInput)
      }
    }
  }

  /**
   * Check if the input map keys are compatible with the model.
   *
   * @param inputMap a map with sub-networks indices as keys
   *
   * @throws IllegalArgumentException if the indices of the map are negative or exceed the last sub-network index
   */
  private fun checkInputMapKeys(inputMap: Map<Int, Any>) {

    inputMap.keys.forEach {
      require(it in 0 until this.model.networks.size) {
        "Index %d not in range [0, %d]".format(it, this.model.networks.lastIndex)
      }
    }
  }

  /**
   * Check if the input map keys are compatible with the networks used for the last scoring.
   *
   * @param inputMap a map with sub-networks indices as keys
   *
   * @throws IllegalArgumentException if the indices of the map are not within the ones used for the last scoring
   */
  private fun checkInputMapUsedKeys(inputMap: Map<Int, Any>) {

    inputMap.keys.forEach {
      require(it in this.usedProcessorsPerNetwork) { "Network $it not used" }
    }
  }

  /**
   * Check if the prediction indices of an input map are compatible with the last predictions done.
   *
   * @param inputMap a map of sub-networks indices to maps of prediction indices to objects
   *
   * @throws IllegalArgumentException if a prediction index is not compatible with the last predictions done
   */
  private fun checkInputMapPredictionIndices(inputMap: Map<Int, Map<Int, Any>>) {

    inputMap.forEach { networkIndex, predictionsMap ->
      predictionsMap.keys.forEach { predictionIndex ->

        require(predictionIndex in 0 until this.usedProcessorsPerNetwork[networkIndex]!!.size ) {
          "%d predictions done with the network %d, but %d given as prediction index"
            .format(networkIndex, this.usedProcessorsPerNetwork[networkIndex]!!.size, predictionIndex)
        }
      }
    }
  }

  /**
   * Initialize processors.
   *
   * @param featuresMap the input features map
   */
  private fun initProcessors(featuresMap: Map<Int, List<InputNDArrayType>>) {

    this.processorsPools.forEach { it.releaseAll() }

    this.usedProcessorsPerNetwork.clear()

    featuresMap.forEach { (networkIndex, processorsList) ->
      this.usedProcessorsPerNetwork[networkIndex] = List(
        size = processorsList.size,
        init = { this.processorsPools[networkIndex].getItem() }
      )
    }
  }

  /**
   * Map each used processor of each sub-network with the given [transform] function.
   *
   * @param transform the transform function to apply to each neural processor
   *
   * @return a map of network indices to lists of objects returned by the [transform] function, one for each used
   *         processor
   */
  private fun <T> mapUsedProcessors(
    transform: (networkIndex: Int, processorIndex: Int, processor: FeedforwardNeuralProcessor<InputNDArrayType>) -> T
  ): Map<Int, List<T>> {

    val networkIndices: List<Int> = this.usedProcessorsPerNetwork.keys.toList()

    return mapOf(*Array(
      size = networkIndices.size,
      init = { i ->
        val networkIndex: Int = networkIndices[i]
        val processorsList = this.usedProcessorsPerNetwork[networkIndex]!!

        Pair(
          networkIndex,
          List(size = processorsList.size, init = { j -> transform(networkIndex, j, processorsList[j]) })
        )
      }
    ))
  }
}
