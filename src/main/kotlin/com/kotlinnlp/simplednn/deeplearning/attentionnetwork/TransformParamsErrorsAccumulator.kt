/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.layers.LayerParametersFactory
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters

/**
 * The accumulator of the errors of the attention-transform-layer parameters.
 *
 * @property inputSize the input size of the attention network
 * @property attentionSize the attention size of the network
 * @property sparseInput whether the input of the network is sparse
 */
class TransformParamsErrorsAccumulator(
  val inputSize: Int,
  val attentionSize: Int,
  val sparseInput: Boolean
) {

  /**
   * A counter of times when errors were accumulated.
   */
  var count = 0
    private set

  /**
   * A boolean indicating whether errors are accumulated.
   */
  private val isEmpty: Boolean get() = this.count == 0

  /**
   * The structure in which to accumulate the errors of the attention-transform-layer parameters.
   */
  private val paramsErrors: FeedforwardLayerParameters = this.paramsErrorsFactory()

  /**
   * @return the errors of the transform layer parameters
   */
  fun getParamsErrors() = if (this.isEmpty) this.paramsErrorsFactory() else this.paramsErrors

  /**
   * Reset the accumulated errors.
   */
  fun reset() {
    this.count = 0
  }

  /**
   * Divide the accumulated errors by the number of accumulations.
   */
  fun averageErrors() {

    if (this.count != 0) {
      val countDouble = this.count.toDouble()
      this.paramsErrors.forEach { it.values.assignDiv(countDouble) }
    }
  }

  /**
   * Accumulate the given [paramsErrors] into the accumulator.
   *
   * @param paramsErrors the attention-transform-layer parameters errors to accumulate
   */
  fun accumulate(paramsErrors: FeedforwardLayerParameters) {
    require(paramsErrors.inputSize == this.inputSize)
    require(paramsErrors.outputSize == this.attentionSize)

    if (this.isEmpty) {
      this.assignValues(paramsErrors)
    } else {
      this.assignSum(paramsErrors)
    }

    this.count += 1
  }

  /**
   * @return the structure in which to accumulate the errors of the attention-transform-layer parameters
   */
  private fun paramsErrorsFactory(): FeedforwardLayerParameters {

    return LayerParametersFactory(
      inputSize = this.inputSize,
      outputSize = this.attentionSize,
      connectionType = LayerType.Connection.Feedforward,
      sparseInput = this.sparseInput
    ) as FeedforwardLayerParameters
  }

  /**
   * Assign the values of the given [paramsErrors] to the ones of the accumulator.
   *
   * @param paramsErrors the parameters errors to assign
   */
  private fun assignValues(paramsErrors: FeedforwardLayerParameters) {

    this.paramsErrors.zip(paramsErrors).forEach { (a, b) -> a.values.assignValues(b.values) }
  }

  /**
   * Add the values of the given [paramsErrors] to the ones of the accumulator.
   *
   * @param paramsErrors the parameters errors to add
   */
  private fun assignSum(paramsErrors: FeedforwardLayerParameters) {

    this.paramsErrors.zip(paramsErrors).forEach { (a, b) -> a.values.assignSum(b.values) }
  }
}
