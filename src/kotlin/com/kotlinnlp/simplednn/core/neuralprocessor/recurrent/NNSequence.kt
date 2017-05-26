/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent


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
  inner class NNState(val structure: RecurrentNetworkStructure<InputNDArrayType>)

  /**
   * Sequence of RNNStates
   */
  val states = ArrayList<NNState>()

  /**
   *
   */
  val paramsErrorsAccumulator: ParamsErrorsAccumulator = ParamsErrorsAccumulator(this.neuralNetwork)

  /**
   *
   * the number of states
   */
  val length: Int
    get() = this.states.size

  /**
   *
   * the last index of the states
   */
  val lastIndex: Int
    get() = this.states.size - 1

  /**
   *
   * @param index index
   * @return
   */
  val isEmpty: Boolean
    get() = this.states.isEmpty()

  /**
   *
   * @param index index
   * @return
   */
  val isNotEmpty: Boolean
    get() = this.states.isNotEmpty()

  /**
   *
   * @param index index
   * @return
   */
  fun isLast(index: Int): Boolean { return index == this.states.size - 1 }

  /**
   *
   * @return
   */
  val lastStructure: RecurrentNetworkStructure<InputNDArrayType>?
    get() = if (this.states.isNotEmpty()) this.states.last().structure else null

  /**
   *
   * @return states last index
   */
  fun getStateStructure(stateIndex: Int): RecurrentNetworkStructure<InputNDArrayType>? {
    return if (stateIndex in 0 .. this.lastIndex) this.states[stateIndex].structure else null
  }

  /**
   *
   */
  fun add(structure: RecurrentNetworkStructure<InputNDArrayType>) = this.states.add(NNState(structure))

  /**
   *
   */
  fun reset() {
    this.states.clear()
    this.paramsErrorsAccumulator.reset()
  }

}
