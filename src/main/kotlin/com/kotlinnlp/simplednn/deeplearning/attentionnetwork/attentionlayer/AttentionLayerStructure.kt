/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The structure of the Attention Layer.
 *
 * @property inputSequence the sequence of input arrays
 * @property attentionSequence the sequence of attention arrays
 * @property params the parameters of the Attention Layer
 */
class AttentionLayerStructure<InputNDArrayType: NDArray<InputNDArrayType>>(
  val inputSequence: ArrayList<AugmentedArray<InputNDArrayType>>,
  attentionSequence: ArrayList<DenseNDArray>,
  params: AttentionParameters
) : AttentionStructure(attentionSequence = attentionSequence, params = params) {

  /**
   * The output dense array.
   */
  val outputArray: AugmentedArray<DenseNDArray>

  /**
   * The size of each array of input.
   */
  private val inputSize: Int

  /**
   * Initialize values.
   */
  init {
    require(this.inputSequence.size > 0) { "The input sequence cannot be empty." }
    require(this.inputSequence.size == this.attentionSequence.size) {
      "The input sequence must have the same length of the attention sequence."
    }

    this.inputSize = this.inputSequence.first().size
    this.checkInputArraysSize() // call it after the inputSize is been set
    this.outputArray = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.inputSize)))
  }

  /**
   * Set the errors of the [outputArray].
   *
   * @param errors the errors to set into the outputArray
   */
  fun setErrors(errors: DenseNDArray) = this.outputArray.assignErrors(errors)

  /**
   * Perform the forward of the input sequence.
   *
   * @return the output array
   */
  fun forward(): DenseNDArray {

    val helper = AttentionLayerForwardHelper(layer = this)

    helper.forward()

    return this.outputArray.values
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input sequence
   */
  fun backward(paramsErrors: AttentionParameters, propagateToInput: Boolean) {

    val helper = AttentionLayerBackwardHelper(layer = this)

    helper.backward(paramsErrors = paramsErrors, propagateToInput = propagateToInput)
  }

  /**
   * Check the size of the input arrays.
   */
  private fun checkInputArraysSize() {
    this.inputSequence.forEach {
      require(it.size == this.inputSize)
    }
  }
}
