/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionMechanism
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The backward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class BackwardHelper(private val network: PointerNetwork) {

  /**
   * The list of errors of the input sequence.
   */
  internal lateinit var inputSequenceErrors: List<DenseNDArray>
    private set

  /**
   * The list of errors of the input vectors.
   */
  internal lateinit var vectorsErrors: List<DenseNDArray>
    private set

  /**
   * The index of the current state (the backward processes the states in inverted order).
   */
  private var stateIndex: Int = 0

  /**
   * The params errors accumulator of the transform vectors.
   */
  private var transformErrorsAccumulator = ParamsErrorsAccumulator<AffineLayerParameters>()

  /**
   * The params errors accumulator of the attention structure
   */
  private var attentionErrorsAccumulator = ParamsErrorsAccumulator<AttentionParameters>()

  /**
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var transformLayerParamsErrors: AffineLayerParameters

  /**
   * The structure used to store the params errors of the attention during the backward.
   */
  private lateinit var attentionParamsErrors: AttentionParameters

  /**
   * Perform the back-propagation from the output errors.
   *
   * @param outputErrors the errors to propagate
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    this.initBackward()

    (0 until outputErrors.size).reversed().forEach { stateIndex ->

      this.stateIndex = stateIndex

      this.backwardStep(outputErrors[stateIndex])
    }

    this.transformErrorsAccumulator.averageErrors()
    this.attentionErrorsAccumulator.averageErrors()
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of the [network]
   */
  fun getParamsErrors(copy: Boolean = true) = PointerNetworkParameters(
    transformParams = this.transformErrorsAccumulator.getParamsErrors(copy = copy),
    attentionParams = this.attentionErrorsAccumulator.getParamsErrors(copy = copy))

  /**
   * A single step of backward.
   *
   * @param outputErrors the errors of a single output array
   */
  private fun backwardStep(outputErrors: DenseNDArray) {

    val attentionArraysErrors: Array<DenseNDArray> = this.backwardAttentionScores(outputErrors)
    val vectorErrors: DenseNDArray = this.backwardAttentionArrays(attentionArraysErrors)

    this.vectorsErrors[this.stateIndex].assignValues(vectorErrors)
  }

  /**
   * @param outputErrors the errors of a single output array
   *
   * @return the errors of the attention arrays
   */
  private fun backwardAttentionScores(outputErrors: DenseNDArray): Array<DenseNDArray> {

    val attentionStructure: AttentionStructure = this.network.forwardHelper.usedAttentionStructures[this.stateIndex]

    AttentionMechanism(attentionStructure).backward(
      paramsErrors = this.getAttentionParamsErrors(),
      importanceScoreErrors = outputErrors)

    this.attentionErrorsAccumulator.accumulate(this.attentionParamsErrors)

    return attentionStructure.getAttentionErrors()
  }

  /**
   *
   */
  private fun backwardAttentionArrays(outputErrors: Array<DenseNDArray>): DenseNDArray {

    val vectorErrorsSum: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.network.model.inputSize))

    val transformLayers: List<AffineLayerStructure<DenseNDArray>>
      = this.network.forwardHelper.usedTransformLayers[this.stateIndex]

    transformLayers.zip(outputErrors).forEachIndexed { index, (transformLayer, attentionErrors) ->

      val (inputSequenceElementError: DenseNDArray, vectorErrors: DenseNDArray) =
        this.backwardTransformLayer(layer = transformLayer, outputErrors = attentionErrors)

      vectorErrorsSum.assignSum(vectorErrors)

      this.inputSequenceErrors[index].assignSum(inputSequenceElementError)
    }

    return vectorErrorsSum
  }


  /**
   * A single transform layer backward.
   *
   * @param layer a transform layer
   * @param outputErrors the errors of the output
   *
   * @return the errors of the input
   */
  private fun backwardTransformLayer(layer: AffineLayerStructure<DenseNDArray>,
                                     outputErrors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val paramsErrors = this.getTransformParamsErrors()

    layer.setErrors(outputErrors)
    layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.transformErrorsAccumulator.accumulate(paramsErrors)

    return layer.getInputErrors(copy = true)
  }

  /**
   * Initialize the structures used during a backward.
   */
  private fun initBackward() {

    this.initInputSequenceErrors()
    this.initVectorsErrors()

    this.transformErrorsAccumulator.reset()
    this.attentionErrorsAccumulator.reset()
  }

  /**
   * Initialize the [inputSequenceErrors] with arrays of zeros (an amount equal to the size of the current input
   * sequence).
   */
  private fun initInputSequenceErrors() {
    this.inputSequenceErrors = List(
      size = this.network.inputSequence.size,
      init = { DenseNDArrayFactory.zeros(Shape(this.network.model.inputSize)) })
  }

  /**
   * Initialize the [vectorsErrors] with arrays of zeros (an amount equal to the size of the number of
   * performed forward).
   */
  private fun initVectorsErrors() {
    this.vectorsErrors = List(
      size = this.network.forwardCount,
      init = { DenseNDArrayFactory.zeros(Shape(this.network.model.vectorSize)) })
  }

  /**
   * @return the transform layers params errors
   */
  private fun getTransformParamsErrors(): AffineLayerParameters {

    if (!this::transformLayerParamsErrors.isInitialized) {
      this.transformLayerParamsErrors = this.network.forwardHelper.usedTransformLayers.last().last().params.copy()
    }

    return this.transformLayerParamsErrors
  }

  /**
   * @return the attention params errors
   */
  private fun getAttentionParamsErrors(): AttentionParameters {

    if (!this::attentionParamsErrors.isInitialized) {
      this.attentionParamsErrors = this.network.forwardHelper.usedAttentionStructures.last().params.copy()
    }

    return this.attentionParamsErrors
  }
}
