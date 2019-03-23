/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [NeuralProcessor]s which allows to allocate and release processors when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a processor is created.
 *
 * @param model the stacked-layers parameters
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @property paramsErrorsCollector where to collect the local params errors during the [backward] (optional)
 */
class FeedforwardNeuralProcessorsPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  private val model: StackedLayersParameters,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean,
  private val paramsErrorsCollector: ParamsErrorsCollector = ParamsErrorsCollector()
  ) : ItemsPool<FeedforwardNeuralProcessor<InputNDArrayType>>() {

  /**
   * The factory of a new processor
   *
   * @param id the id of the processor to create
   *
   * @return a new [FeedforwardNeuralProcessor] with the given [id]
   */
  override fun itemFactory(id: Int) = FeedforwardNeuralProcessor<InputNDArrayType>(
    model = this.model,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput,
    paramsErrorsCollector = this.paramsErrorsCollector,
    id = id
  )
}
