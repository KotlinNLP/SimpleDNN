/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [NeuralProcessor]s which allows to allocate and release processors when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a processor is created.
 *
 * @property neuralNetwork the [NeuralNetwork] which the processors of the pool will work with
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @param mePropK a list of k factors (one per layer) of the 'meProp' algorithm to propagate from the k (in
 *                percentage) output nodes with the top errors of each layer (the list and each element can be null)
 */
class RecurrentNeuralProcessorsPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val neuralNetwork: NeuralNetwork,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean,
  private val mePropK: List<Double?>? = null
) : ItemsPool<RecurrentNeuralProcessor<InputNDArrayType>>() {

  /**
   * The factory of a new processor
   *
   * @param id the id of the processor to create
   *
   * @return a new [RecurrentNeuralProcessor] with the given [id]
   */
  override fun itemFactory(id: Int) = RecurrentNeuralProcessor<InputNDArrayType>(
    neuralNetwork = this.neuralNetwork,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput,
    mePropK = this.mePropK,
    id = id
  )
}
