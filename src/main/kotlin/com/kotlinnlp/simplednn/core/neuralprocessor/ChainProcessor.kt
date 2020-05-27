/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList

/**
 * The ChainProcessor can be useful when you intend to use the output of one processor as the input of another
 * directly or passing through an arbitrary number of intermediate processors.
 *
 * The 'generics' ensure type compatibility, but each processor is required to check the compatibility of input,
 * output and error sizes.
 *
 * @param inputProcessor the input processor
 * @param hiddenProcessors the hidden processors (can be an empty list)
 * @param outputProcessor the output processor
 * @property id an identification number of this processor
 */
class ChainProcessor<
  in InputType : Any,
  out OutputType : Any,
  in ErrorsType : Any,
  out InputErrorsType : Any,
  HiddenIOType: Any
  > (
  val inputProcessor: NeuralProcessor<InputType, HiddenIOType, HiddenIOType, InputErrorsType>,
  val hiddenProcessors: List<NeuralProcessor<HiddenIOType, HiddenIOType, HiddenIOType, HiddenIOType>>,
  val outputProcessor: NeuralProcessor<HiddenIOType, OutputType, ErrorsType, HiddenIOType>,
  override val id: Int = 0
) : NeuralProcessor<
  InputType, // InputType
  OutputType, // OutputType
  ErrorsType, // ErrorsType
  InputErrorsType // InputErrorsType
  > {

  /**
   * Whether to propagate the errors to the input during the [backward] (if supported)
   */
  override val propagateToInput: Boolean = this.inputProcessor.propagateToInput

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: InputType): OutputType =
    this.outputProcessor.forward(
      this.hiddenProcessors.forward(
        this.inputProcessor.forward(input)))

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: ErrorsType) {

    this.inputProcessor.backward(
      this.hiddenProcessors.backwardAndGetInputErrors(
        this.outputProcessor.let {
          it.backward(outputErrors)
          it.getInputErrors(copy = false)
        }
      )
    )
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.inputProcessor.getParamsErrors(copy) +
      this.hiddenProcessors.getParamsErrors(copy) +
      this.outputProcessor.getParamsErrors(copy)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): InputErrorsType = this.inputProcessor.getInputErrors(copy)

  /**
   * Perform the forward of the hidden processors.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  private fun List<NeuralProcessor<
    HiddenIOType, HiddenIOType, HiddenIOType, HiddenIOType
    >>.forward(input: HiddenIOType): HiddenIOType {

    var curInput = input

    this.forEach { curInput = it.forward(curInput) }

    return curInput
  }

  /**
   * Perform the backward of the hidden processors returning the input errors.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors of the first hidden processor, or the [outputErrors] if there are no hidden processors
   */
  private fun List<NeuralProcessor<
    HiddenIOType, HiddenIOType, HiddenIOType, HiddenIOType
    >>.backwardAndGetInputErrors(outputErrors: HiddenIOType): HiddenIOType {

    var errors = outputErrors

    this.asReversed().forEach { proc ->

      errors = proc.let {
        it.backward(errors)
        it.getInputErrors(copy = false)
      }
    }

    return errors
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  private fun List<NeuralProcessor<*, *, *, *>>.getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.map { it.getParamsErrors(copy) }.flatten()
}
