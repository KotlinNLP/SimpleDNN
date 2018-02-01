/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.utils.ItemsPool

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
 * @property id an identification number useful to track a specific [BiRNNEncoder]
 */
class BiRNNEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val network: BiRNN,
  override val id: Int = 0
) : ItemsPool.IDItem {

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence left-to-right.
   */
  private val leftToRightProcessor = RecurrentNeuralProcessor<InputNDArrayType>(this.network.leftToRightNetwork)

  /**
   * The [RecurrentNeuralProcessor] which encodes the sequence right-to-left.
   */
  private val rightToLeftProcessor = RecurrentNeuralProcessor<InputNDArrayType>(this.network.rightToLeftNetwork)

  /**
   * The input sequence.
   */
  private lateinit var sequence: Array<InputNDArrayType>

  /**
   * Encode the [sequence].
   *
   * @param sequence the sequence to encode
   * @param useDropout whether to apply the dropout
   *
   * @return the encoded sequence
   */
  fun encode(sequence: Array<InputNDArrayType>, useDropout: Boolean = false): Array<DenseNDArray> {

    this.sequence = sequence

    val (leftToRightOut, rightToLeftOut) = this.biEncoding(useDropout = useDropout)

    return BiRNNUtils.concatenate(leftToRightOut, rightToLeftOut)
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
   * This method should be called only after an [encode] call.
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
   * @return the concatenation of the final output arrays of the two RNNs (left-to-right + right-to-left)
   */
  fun getMixedFinalOutput(): DenseNDArray = concatVectorsV(
    this.leftToRightProcessor.getOutput(copy = false),
    this.rightToLeftProcessor.getOutput(copy = false)
  )

  /**
   * Propagate the errors of the entire sequence.
   *
   * @param outputErrorsSequence the errors to propagate
   * @param propagateToInput whether to propagate the output errors to the input or not
   */
  fun backward(outputErrorsSequence: Array<DenseNDArray>, propagateToInput: Boolean) {

    val (leftToRightOutputErrors, rightToLeftOutputErrors) =
      BiRNNUtils.splitErrorsSequence(outputErrorsSequence)

    this.leftToRightProcessor.backward(
      outputErrorsSequence = leftToRightOutputErrors,
      propagateToInput = propagateToInput)

    this.rightToLeftProcessor.backward(
      outputErrorsSequence = rightToLeftOutputErrors.reversed().toTypedArray(),
      propagateToInput = propagateToInput)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence (the errors of the two RNNs are combined by summation)
   */
  fun getInputSequenceErrors(copy: Boolean = true): Array<DenseNDArray> {
    return BiRNNUtils.sumBidirectionalErrors(
      leftToRightInputErrors = this.leftToRightProcessor.getInputSequenceErrors(copy = copy),
      rightToLeftInputErrors = this.rightToLeftProcessor.getInputSequenceErrors(copy = copy)
    )
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the BiRNN parameters
   */
  fun getParamsErrors(copy: Boolean = true): BiRNNParameters {
    return BiRNNParameters(
      leftToRight = leftToRightProcessor.getParamsErrors(copy = copy),
      rightToLeft = rightToLeftProcessor.getParamsErrors(copy = copy)
    )
  }

  /**
   * Given a [sequence] return the encoded left-to-right and right-to-left representation.
   *
   * @param useDropout whether to apply the dropout
   *
   * @return a Pair with two arrays containing the outputs of the two RNNs
   */
  private fun biEncoding(useDropout: Boolean):
    Pair<Array<DenseNDArray>, Array<DenseNDArray>> {

    val leftToRightOut = arrayOfNulls<DenseNDArray>(this.sequence.size)
    val rightToLeftOut = arrayOfNulls<DenseNDArray>(this.sequence.size)

    var isFirstElement = true

    this.sequence.indices.zip(this.sequence.indices.reversed()).forEach { (i, r) ->
      leftToRightOut[i] = this.leftToRightProcessor.forward(
        featuresArray = this.sequence[i],
        firstState = isFirstElement,
        useDropout = useDropout)
      rightToLeftOut[r] = this.rightToLeftProcessor.forward(
        featuresArray = this.sequence[r],
        firstState = isFirstElement,
        useDropout = useDropout)

      isFirstElement = false
    }

    return Pair(
      leftToRightOut.requireNoNulls(),
      rightToLeftOut.requireNoNulls()
    )
  }
}
