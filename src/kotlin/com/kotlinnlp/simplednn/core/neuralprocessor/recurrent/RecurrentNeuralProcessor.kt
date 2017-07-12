/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.GatedRecurrentLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.RecurrentLayerStructure
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.RecurrentNetworkStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.StructureContextWindow
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The [RecurrentNeuralProcessor] acts on the [neuralNetwork] performing predictions
 * and training based on sequences of recurrent Examples.
 *
 * @property neuralNetwork a [NeuralNetwork]
 */
class RecurrentNeuralProcessor<InputNDArrayType : NDArray<InputNDArrayType>>(
  neuralNetwork: NeuralNetwork
) : StructureContextWindow<InputNDArrayType>,
    NeuralProcessor(neuralNetwork) {

  /**
   * Sequence of states.
   */
  private val sequence = NNSequence<InputNDArrayType>(neuralNetwork)

  /**
   * Set each time a single forward or a single backward are called
   */
  private var curStateIndex: Int = 0

  /**
   * An index which indicates the last state (-1 if the sequence is empty).
   */
  private var lastStateIndex: Int = -1

  /**
   * The errors of the network model parameters calculated during a single backward
   */
  private var backwardParamsErrors: NetworkParameters = this.neuralNetwork.parametersErrorsFactory()

  /**
   *
   */
  private var paramsErrorsAccumulator: ParamsErrorsAccumulator = ParamsErrorsAccumulator(neuralNetwork)

  /**
   *
   */
  private val zeroErrors: DenseNDArray = DenseNDArrayFactory.zeros(
    Shape(this.neuralNetwork.layersConfiguration.last().size)
  )

  /**
   *
   */
  override fun getPrevStateStructure(): RecurrentNetworkStructure<InputNDArrayType>? {
    return if (this.curStateIndex in 1 .. this.lastStateIndex)
      this.sequence.getStateStructure(this.curStateIndex - 1)
    else
      null
  }

  /**
   *
   */
  override fun getNextStateStructure(): RecurrentNetworkStructure<InputNDArrayType>? {
    return if (this.curStateIndex in 0 until this.lastStateIndex)
      this.sequence.getStateStructure(this.curStateIndex + 1)
    else
      null
  }

  /**
   *
   */
  override fun getParamsErrors(copy: Boolean): NetworkParameters {

    val paramsError: NetworkParameters

    if (copy) {
      paramsError = this.neuralNetwork.parametersErrorsFactory()
      paramsError.assignValues(this.paramsErrorsAccumulator.getParamsErrors())

    } else {
      paramsError = this.paramsErrorsAccumulator.getParamsErrors()
    }

    return paramsError
  }

  /**
   *
   */
  override fun getOutput(copy: Boolean): DenseNDArray {
    return if (copy) {
      this.sequence.getStateStructure(this.lastStateIndex).outputLayer.outputArray.values.copy()
    } else {
      this.sequence.getStateStructure(this.lastStateIndex).outputLayer.outputArray.values
    }
  }

  /**
   *
   */
  fun getInputSequenceErrors(copy: Boolean = true) = Array(
    size = this.sequence.length,
    init = { i ->
      val inputErrors = this.sequence.getStateStructure(i).inputLayer.inputArray.errors

      require(inputErrors is DenseNDArray) {
        "Input errors available only if input is dense"
      }

      if (copy) {
        inputErrors.copy()
      } else {
        inputErrors
      }
    }
  )

  /**
   *
   */
  fun getOutputSequence(copy: Boolean = true): Array<DenseNDArray> =
    Array(size = this.sequence.length, init = { i ->
      if (copy) {
        this.sequence.getStateStructure(i).outputLayer.outputArray.values.copy()
      } else {
        this.sequence.getStateStructure(i).outputLayer.outputArray.values
      }
    })

  /**
   * Forward a sequence.
   *
   * @param sequenceFeaturesArray the features to forward for each item of the sequence
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param useDropout whether to apply the dropout
   *
   * @return the last output of the network after the whole sequence is been forwarded
   */
  fun forward(sequenceFeaturesArray: ArrayList<InputNDArrayType>,
              saveContributions: Boolean = false,
              useDropout: Boolean = false): DenseNDArray {

    sequenceFeaturesArray.forEachIndexed { i, features ->
      this.forward(
        featuresArray = features,
        firstState = (i == 0),
        saveContributions = saveContributions,
        useDropout = useDropout)
    }

    return this.getOutput()
  }

  /**
   * Forward features.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate
   *                          the relevance)
   * @param firstState whether the current one is the first state
   * @param useDropout whether to apply the dropout
   */
  fun forward(featuresArray: InputNDArrayType,
              firstState: Boolean,
              saveContributions: Boolean = false,
              useDropout: Boolean = false): DenseNDArray {

    if (firstState) {
      this.reset()
    }

    this.addNewState(saveContributions = saveContributions)

    this.curStateIndex = this.lastStateIndex

    this.forwardCurrentState(
      featuresArray = featuresArray,
      saveContributions = saveContributions,
      useDropout = useDropout)

    return this.getOutput()
  }

  /**
   * Calculate the relevance of the input of a state respect of the output of another (or the same) state.
   *
   * @param stateFrom the index of the state from whose input to calculate the relevance
   * @param stateTo the index of the state whose output will be used as reference to calculate the relevance
   * @param relevantOutcomesDistribution the distribution which indicates which outcomes are relevant, used
   *                                     as reference to calculate the relevance of the input
   * @param copy whether to return a copy of the relevance or not
   *
   * @return the relevance of the input of the state [stateFrom] respect of the output of the state [stateTo]
   */
  fun calculateRelevance(stateFrom: Int,
                         stateTo: Int,
                         relevantOutcomesDistribution: DistributionArray,
                         copy: Boolean = true): NDArray<*> {

    require(stateFrom <= stateTo) { "stateFrom (%d) must be <= stateTo (%d)".format(stateFrom, stateTo) }
    require(stateFrom in 0 until this.sequence.length) {
      "stateFrom (%d) index exceeded sequence size (%d)".format(stateFrom, this.sequence.length)
    }

    this.sequence.getStateStructure(stateTo).layers.last().setOutputRelevance(relevantOutcomesDistribution)

    for (stateIndex in (stateFrom .. stateTo).reversed()) {
      this.curStateIndex = stateIndex // crucial to provide the right context
      this.propagateRelevanceOnCurrentState(isFirstState = stateIndex == stateFrom, isLastState = stateIndex == stateTo)
    }

    return this.getInputRelevance(stateIndex = stateFrom, copy = copy)
  }

  /**
   * Backward errors.
   *
   * @param outputErrors the errors of the output
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrors: DenseNDArray, propagateToInput: Boolean = false) {

    val outputErrorsSequence = Array(
      size = this.sequence.length,
      init = { i -> if (i == this.lastStateIndex) outputErrors else this.zeroErrors })

    this.backward(outputErrorsSequence = outputErrorsSequence, propagateToInput = propagateToInput)
  }

  /**
   * Backward errors of a sequence.
   *
   * @param outputErrorsSequence output errors for each item of the sequence
   * @param propagateToInput whether to propagate the errors to the input
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean = false) {

    require(outputErrorsSequence.size == this.sequence.length) {
      "Number of errors (${outputErrorsSequence.size}) does not " +
        "reflect the length of the sequence (${this.sequence.length})"
    }

    for ((i, state) in this.sequence.states.withIndex().reversed()) {

      this.curStateIndex = i // crucial to provide the right context

      state.structure.backward(
        outputErrors = outputErrorsSequence[i],
        paramsErrors = this.backwardParamsErrors,
        propagateToInput = propagateToInput)

      this.paramsErrorsAccumulator.accumulate(this.backwardParamsErrors)
    }

    this.paramsErrorsAccumulator.averageErrors()
  }

  /**
   * Add a new state.
   *
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate relevance)
   */
  private fun addNewState(saveContributions: Boolean = false) {

    // TODO: save always contributions?? (structures are created only when it's needed)

    if (this.lastStateIndex == this.sequence.lastIndex) {

      val structure = RecurrentNetworkStructure(
        layersConfiguration = this.neuralNetwork.layersConfiguration,
        params = this.neuralNetwork.model,
        structureContextWindow = this)

      this.sequence.add(structure = structure, saveContributions = saveContributions)
    }

    this.lastStateIndex++
  }

  /**
   * Forward the current state.
   *
   * @param featuresArray the features to forward from the input to the output
   * @param saveContributions whether to save the contributions of each input to its output (needed to calculate relevance)
   * @param useDropout whether to apply the dropout
   */
  private fun forwardCurrentState(
    featuresArray: InputNDArrayType,
    saveContributions: Boolean,
    useDropout: Boolean = false) {

    if (saveContributions) {
      this.sequence.getStateStructure(this.lastStateIndex).forward(
        features = featuresArray,
        networkContributions = this.sequence.getStateContributions(this.lastStateIndex),
        useDropout = useDropout)

    } else {
      this.sequence.getStateStructure(this.lastStateIndex).forward(
        features = featuresArray,
        useDropout = useDropout)
    }
  }

  /**
   * Propagate the relevance backward through the layers of the current state.
   *
   * @param isFirstState a Boolean indicating if the current state is the first
   * @param isLastState a Boolean indicating if the current state is the last
   */
  private fun propagateRelevanceOnCurrentState(isFirstState: Boolean, isLastState: Boolean) {

    val structure: RecurrentNetworkStructure<InputNDArrayType> = this.sequence.getStateStructure(this.curStateIndex)
    var isPropagating: Boolean = isLastState

    for ((layerIndex, layer) in structure.layers.withIndex().reversed()) {

      structure.curLayerIndex = layerIndex // crucial to provide the right context

      val isCurLayerRecurrent = layer is RecurrentLayerStructure
      val isPrevLayerRecurrent = layerIndex > 0 && structure.layers[layerIndex - 1] is RecurrentLayerStructure

      isPropagating = isPropagating || isCurLayerRecurrent

      if (isPropagating) {
        this.propagateLayerRelevance(
          layer = layer,
          layerIndex = layerIndex,
          propagateToPrevState = !isFirstState && isCurLayerRecurrent,
          propagateToInput = layerIndex > 0 || isFirstState,
          replaceInputRelevance = isLastState || !isPrevLayerRecurrent
        )
      }
    }
  }

  /**
   * Propagate the relevance backward through the current layer.
   *
   * @param layer the current layer
   * @param layerIndex the current layer index
   * @param propagateToPrevState whether to propagate to the previous state
   * @param propagateToInput whether to propagate to the input
   * @param replaceInputRelevance a Boolean if the relevance of the input must be replaced or added
   */
  private fun propagateLayerRelevance(layer: LayerStructure<*>,
                                      layerIndex: Int,
                                      propagateToPrevState: Boolean,
                                      propagateToInput: Boolean,
                                      replaceInputRelevance: Boolean) {

    require(propagateToInput || propagateToPrevState)

    val contributions: LayerParameters
      = this.sequence.getStateContributions(this.curStateIndex).paramsPerLayer[layerIndex]

    if (layer is GatedRecurrentLayerStructure) {
      layer.propagateRelevanceToGates(layerContributions = contributions)
    }

    if (propagateToInput) {

      if (replaceInputRelevance) {
        layer.setInputRelevance(layerContributions = contributions)

      } else {
        layer.addInputRelevance(layerContributions = contributions)
      }
    }

    if (propagateToPrevState) { layer as RecurrentLayerStructure
      layer.setRecurrentRelevance(layerContributions = contributions)
    }
  }

  /**
   * Get the relevance of the input of a state of the sequence.
   * (If the input is Dense it is Dense, if the input is Sparse or SparseBinary it is Sparse).
   *
   * @param stateIndex the index of the state from which to extract the input relevance
   * @param copy whether to return a copy of the relevance or not
   *
   * @return the relevance of the input as [NDArray]
   */
  private fun getInputRelevance(stateIndex: Int, copy: Boolean = true): NDArray<*> {

    return if (copy) {
      this.sequence.getStateStructure(stateIndex).inputLayer.inputArray.relevance.copy()
    } else {
      this.sequence.getStateStructure(stateIndex).inputLayer.inputArray.relevance
    }
  }

  /**
   * Reset the sequence.
   */
  private fun reset() {
    this.lastStateIndex = -1
    this.paramsErrorsAccumulator.reset()
  }
}
