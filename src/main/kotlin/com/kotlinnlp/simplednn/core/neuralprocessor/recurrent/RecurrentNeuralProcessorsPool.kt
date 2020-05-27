/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [NeuralProcessor]s which allows to allocate and release processors when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a processor is created.
 *
 * @property model the stacked-layers parameters
 * @param dropouts the probability of dropout for each stacked layer
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @param paramsErrorsCollector where to collect the local params errors during the backward (optional)
 */
class RecurrentNeuralProcessorsPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: StackedLayersParameters,
  private val dropouts: List<Double>,
  private val propagateToInput: Boolean,
  private val paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector()
) : ItemsPool<RecurrentNeuralProcessor<InputNDArrayType>>() {

  /**
   * The neural processor that acts on networks of stacked-layers with recurrent connections.
   *
   * @param model the stacked-layers parameters
   * @param dropout the probability of dropout for each stacked layer (default 0.0)
   * @param propagateToInput whether to propagate the errors to the input during the backward
   * @param paramsErrorsCollector where to collect the local params errors during the backward (optional)
   */
  constructor(
    model: StackedLayersParameters,
    dropout: Double = 0.0,
    propagateToInput: Boolean,
    paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector()
  ): this(
    model = model,
    dropouts = List(model.numOfLayers) { dropout },
    propagateToInput = propagateToInput,
    paramsErrorsCollector = paramsErrorsCollector
  )

  /**
   * The factory of a new processor
   *
   * @param id the id of the processor to create
   *
   * @return a new [RecurrentNeuralProcessor] with the given [id]
   */
  override fun itemFactory(id: Int) = RecurrentNeuralProcessor<InputNDArrayType>(
    model = this.model,
    dropouts = this.dropouts,
    propagateToInput = this.propagateToInput,
    paramsErrorsCollector = this.paramsErrorsCollector,
    id = id
  )
}
