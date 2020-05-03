/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concatff

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters

/**
 * The parameters of the Concat Feedforward layer.
 *
 * @property inputsSize the size of each input
 * @property outputSize the size of the output
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class ConcatFFLayerParameters(
  inputsSize: List<Int>,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : MergeLayerParameters(
  inputsSize = inputsSize,
  outputSize = outputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer,
  sparseInput = false // actually not used because non-dense concatenation is not available
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The output params.
   */
  val output = FeedforwardLayerParameters(
    inputSize = inputsSize.sum(),
    outputSize = outputSize,
    sparseInput = this.sparseInput)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = this.output.weightsList

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = this.output.biasesList

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }
}
