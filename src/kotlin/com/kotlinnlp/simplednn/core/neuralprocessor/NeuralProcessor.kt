/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [NeuralProcessor] acts on the [neuralNetwork] performing predictions and training.
 *
 * @property neuralNetwork a [NeuralNetwork]
 */
abstract class NeuralProcessor(val neuralNetwork: NeuralNetwork) {

  /**
   *
   */
  protected val inputType = this.neuralNetwork.layersConfiguration.first().inputType

  /**
   *
   */
  abstract fun getParamsErrors(copy: Boolean = true): NetworkParameters

  /**
   *
   */
  abstract fun getOutput(copy: Boolean = true): DenseNDArray
}
