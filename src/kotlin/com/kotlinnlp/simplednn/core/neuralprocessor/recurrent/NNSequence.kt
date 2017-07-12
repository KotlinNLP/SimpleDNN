/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent


import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.recurrent.*
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 *
 */
class NNSequence<InputNDArrayType : NDArray<InputNDArrayType>>(val neuralNetwork: NeuralNetwork) {

  /**
   *
   * @param structure neural network
   */
  inner class NNState(
    val structure: RecurrentNetworkStructure<InputNDArrayType>,
    val contributions: NetworkParameters? = null
  )

  /**
   * Sequence of RNNStates
   */
  val states = ArrayList<NNState>()

  /**
   *
   */
  val paramsErrorsAccumulator: ParamsErrorsAccumulator = ParamsErrorsAccumulator(this.neuralNetwork)

  /**
   * The number of states.
   */
  val length: Int
    get() = this.states.size

  /**
   * The last index of the sequence.
   */
  val lastIndex: Int
    get() = this.states.size - 1

  /**
   * The structure of the last state of the sequence. It requires that the sequence is not empty.
   */
  val lastStructure: RecurrentNetworkStructure<InputNDArrayType>? = this.getStateStructure(this.lastIndex)

  /**
   * The contributions of the last state of the sequence. It requires that the sequence is not empty.
   */
  val lastContributions: NetworkParameters get() = this.getStateContributions(this.lastIndex)!!

  /**
   * Get the structure of the state at the given [stateIndex].
   *
   * @param stateIndex the index of the sequence
   *
   * @return the structure of the state at the given [stateIndex] or null if the [stateIndex] exceeds the length
   */
  fun getStateStructure(stateIndex: Int): RecurrentNetworkStructure<InputNDArrayType>? {
    return if (stateIndex in 0..this.lastIndex) this.states[stateIndex].structure else null
  }

  /**
   * Get the contributions of the state at the given [stateIndex].
   *
   * @param stateIndex the index of the sequence
   *
   * @return the contributions of the state at the given [stateIndex] or null if the [stateIndex] exceeds the length
   */
  fun getStateContributions(stateIndex: Int): NetworkParameters? {
    return if (stateIndex in 0..this.lastIndex) this.states[stateIndex].contributions else null
  }

  /**
   * Add a new state to the sequence.
   *
   * @param structure the [RecurrentNetworkStructure] of the state
   * @param saveContributions whether to include the contributions structure into the state
   */
  fun add(structure: RecurrentNetworkStructure<InputNDArrayType>, saveContributions: Boolean) {

    this.states.add(
      NNState(
        structure = structure,
        contributions = if (saveContributions) this.neuralNetwork.parametersErrorsFactory() else null
      )
    )
  }

  /**
   * Reset the sequence.
   */
  fun reset() {
    this.states.clear()
    this.paramsErrorsAccumulator.reset()
  }
}
