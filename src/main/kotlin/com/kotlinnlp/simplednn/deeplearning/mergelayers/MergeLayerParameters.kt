package com.kotlinnlp.simplednn.deeplearning.mergelayers

import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of a merge layer.
 * It has two inputs instead of one.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
abstract class MergeLayerParameters(
  val inputSize1: Int,
  val inputSize2: Int,
  outputSize: Int,
  private val sparseInput: Boolean
) : LayerParameters(
  inputSize = inputSize1,
  outputSize = outputSize
)
