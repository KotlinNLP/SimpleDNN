/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * The [NeuralProcessor] acts on the [neuralNetwork] performing predictions and training.
 *
 * @property neuralNetwork a [NeuralNetwork]
 * @property id an identification number useful to track a specific processor
 */
abstract class NeuralProcessor(val neuralNetwork: NeuralNetwork, override val id: Int = 0) : ItemsPool.IDItem {

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the parameters
   */
  abstract fun getParamsErrors(copy: Boolean = true): NetworkParameters
}
