/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The neural processor that acts on more networks of stacked-layers, performing operations with the same input batch
 * and obtaining more outputs for each element.
 *
 * It forwards the same sequence of X arrays using N different networks and obtaining N outputs for each element of the
 * sequence.
 *
 * @property model the parameters of more stacked-layers networks
 * @param dropouts the probability of dropout for each stacked layer of each network
 * @property propagateToInput whether to propagate the errors to the input during the backward
 * @property id an identification number useful to track a specific processor
 */
class MultiBatchFeedforwardProcessor<InputNDArrayType: NDArray<InputNDArrayType>>(
  val model: List<StackedLayersParameters>,
  dropouts: List<List<Double>>,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<List<DenseNDArray>>, // OutputType
  List<List<DenseNDArray>>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * The neural processor that acts on networks of stacked-layers, performing operations through with mini-batches.
   *
   * @property model the parameters of more stacked-layers networks
   * @param dropout the probability of dropout for each stacked layer of each network (default 0.0)
   * @param propagateToInput whether to propagate the errors to the input during the [backward]
   * @param id an identification number useful to track a specific processor
   */
  constructor(
    model: List<StackedLayersParameters>,
    dropout: Double = 0.0,
    propagateToInput: Boolean,
    id: Int = 0
  ): this(
    model = model,
    dropouts = model.map { List(it.numOfLayers) { dropout } },
    propagateToInput = propagateToInput,
    id = id
  )

  /**
   * The feed-forward processors to encode each input batch.
   */
  private val encoders: List<BatchFeedforwardProcessor<InputNDArrayType>> =
    this.model.zip(dropouts).map { (params, processorDropouts) ->
      BatchFeedforwardProcessor<InputNDArrayType>(
        model = params,
        dropouts = processorDropouts,
        propagateToInput = this.propagateToInput)
    }

  /**
   * @param copy whether the returned errors must be a copy or a reference (actually without effect, the errors are
   *             always copied!)
   *
   * @return the errors of the input batch accumulated from all the networks
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {

    val inputErrors: List<DenseNDArray> = this.encoders[0].getInputErrors(copy = true)

    for (encoderIndex in 1 until (this.model.size - 1)) {
      inputErrors.zip(this.encoders[encoderIndex].getInputErrors(copy = false)).forEach {
        (baseErrors, errors) -> baseErrors.assignSum(errors)
      }
    }

    return inputErrors
  }

  /**
   * @param copy whether the returned errors must be a copy or a reference
   *
   * @return the parameters errors of all the networks
   */
  override fun getParamsErrors(copy: Boolean) = this.encoders.flatMap { it.getParamsErrors(copy = copy) }

  /**
   * For each network, execute the forward of the same input batch to the output.
   *
   * @param input the input batch
   *
   * @return the outputs of all the networks for each element of the input batch
   */
  override fun forward(input: List<InputNDArrayType>): List<List<DenseNDArray>> {

    val encodersOutputs: List<List<DenseNDArray>> = this.encoders.map { it.forward(input) }

    return List(
      size = input.size,
      init = { elementIndex ->
        List(size = this.encoders.size, init = { encoderIndex -> encodersOutputs[encoderIndex][elementIndex] })
      }
    )
  }

  /**
   * Execute the backward for each network, given their output errors.
   *
   * @param outputErrors the output errors of each network
   */
  override fun backward(outputErrors: List<List<DenseNDArray>>) {

    this.encoders.forEachIndexed { encoderIndex, encoder ->
      encoder.backward(outputErrors.map { it[encoderIndex] } )
    }
  }
}
