/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * Bidirectional Recursive Neural Network Encoder
 *
 * The BiRNNEncoder encodes a sequence (i.e., words) using a concatenation of two RNNs,
 * one processing from left-to-right and the other processing right-to-left.
 * The result is a vector representation for each element which captures information
 * of the element itself and an “infinite” window around it.
 *
 * This implementation support a sequence encoding at time.
 *
 * @property network the [BiRNN] of this encoder
 * @property useDropout whether to apply the dropout during the [forward]
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property id an identification number useful to track a specific [BiRNNEncoder]
 */
class BiRNNEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val network: BiRNN,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence left-to-right.
   */
  private val leftToRightProcessor = RecurrentNeuralProcessor<InputNDArrayType>(
    model = this.network.leftToRightNetwork,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput)

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence right-to-left.
   */
  private val rightToLeftProcessor = RecurrentNeuralProcessor<InputNDArrayType>(
    model = this.network.rightToLeftNetwork,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput)

  /**
   * The processor that merge the left-to-right and right-to-left encoded vectors.
   */
  private val outputMergeProcessors = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.network.outputMergeNetwork,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The input sequence.
   */
  private lateinit var sequence: List<InputNDArrayType>

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: List<InputNDArrayType>): List<DenseNDArray> {

    this.sequence = input

    return outputMergeProcessors.forward(ArrayList(this.biEncoding().map { it.toList() } ))
  }

  /**
   * Get the RAN importance scores of the previous states (split by left and right) for each element of the last
   * encoded sequence.
   * The scores values are in the range [0.0, 1.0].
   *
   * Both the left and the right scores are given following the linear order of the input sequence.
   * E.g.: for an input sequence of 8 elements, the importance scores of the 4th element are related to the following
   * elements:
   * left -> [elm-1, elm-2, elm-3]
   * right -> [elm-5, elm-6, elm-7, elm-8]
   *
   * This method should be called only after a [forward] call.
   * It is required that the networks structures contain only a RAN layer.
   *
   * @return the list of importance scores pairs (left, right) of the input elements
   */
  fun getRANImportanceScores(): List<Pair<DenseNDArray?, DenseNDArray?>> {

    val statesSize: Int = this.sequence.size

    return (0 until statesSize).map { stateIndex ->
      val leftStateIndex: Int = stateIndex
      val rightStateIndex: Int = statesSize - stateIndex - 1

      Pair(
        if (leftStateIndex > 0) this.leftToRightProcessor.getRANImportanceScores(leftStateIndex) else null,
        if (rightStateIndex > 0) this.rightToLeftProcessor.getRANImportanceScores(rightStateIndex) else null
      )
    }
  }

  /**
   * @param copy whether to return a copy of the arrays
   *
   * @return a pair containing the last output of the two RNNs (left-to-right, right-to-left).
   */
  fun getLastOutput(copy: Boolean): Pair<DenseNDArray, DenseNDArray> = Pair(
    this.leftToRightProcessor.getOutput(copy = copy),
    this.rightToLeftProcessor.getOutput(copy = copy)
  )

  /**
   * Propagate the errors of the last output of the two RNNs (left-to-right, right-to-left).
   *
   * @param leftToRightErrors the last output errors of the left-to-right network
   * @param rightToLeftErrors the last output errors of the right-to-left network
   */
  fun backwardLastOutput(leftToRightErrors: DenseNDArray, rightToLeftErrors: DenseNDArray) {

    this.leftToRightProcessor.backward(leftToRightErrors)
    this.rightToLeftProcessor.backward(rightToLeftErrors)
  }

  /**
   * Propagate the errors of the entire sequence.
   *
   * @param outputErrors the errors to propagate
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    val (leftToRightOutputErrors, rightToLeftOutputErrors) = this.outputMergeBackward(outputErrors).unzip()

    this.leftToRightProcessor.backward(leftToRightOutputErrors)
    this.rightToLeftProcessor.backward(rightToLeftOutputErrors.reversed())
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence (the errors of the two RNNs are combined by summation)
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    BiRNNUtils.sumBidirectionalErrors(
      leftToRightSequenceErrors = this.leftToRightProcessor.getInputErrors(copy = copy),
      rightToLeftSequenceErrors = this.rightToLeftProcessor.getInputErrors(copy = copy)
    )

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the BiRNN parameters
   */
  override fun getParamsErrors(copy: Boolean) =
    this.leftToRightProcessor.getParamsErrors(copy = copy) +
      this.rightToLeftProcessor.getParamsErrors(copy = copy) +
        this.outputMergeProcessors.getParamsErrors(copy = copy)


  /**
   * Get the left-to-right and right-to-left lists containing the encoded vectors of the input [sequence].
   * The i-th vector of each list is the encoding of the i-th element of the input [sequence].
   *
   * @return a list of pairs containing the outputs of the two RNNs
   */
  private fun biEncoding(): List<Pair<DenseNDArray, DenseNDArray>> {

    var isFirstElement = true

    val l2rEncodings: MutableList<DenseNDArray> = mutableListOf()
    val r2lEncodings: MutableList<DenseNDArray> = mutableListOf()

    this.sequence.indices.zip(this.sequence.indices.reversed()).forEach { (i, r) ->

      l2rEncodings.add(
        this.leftToRightProcessor.forward(
          input = this.sequence[i],
          firstState = isFirstElement)
      )

      r2lEncodings.add(
        0, // prepend
        this.rightToLeftProcessor.forward(
          input = this.sequence[r],
          firstState = isFirstElement)
      )

      isFirstElement = false
    }

    return l2rEncodings.zip(r2lEncodings)
  }

  /**
   *
   */
  private fun outputMergeBackward(outputErrorsSequence: List<DenseNDArray>): List<Pair<DenseNDArray, DenseNDArray>> {
    this.outputMergeProcessors.backward(outputErrorsSequence)
    return this.outputMergeProcessors.getInputsErrors(copy = false).map { Pair(it[0], it[1]) }
  }
}
