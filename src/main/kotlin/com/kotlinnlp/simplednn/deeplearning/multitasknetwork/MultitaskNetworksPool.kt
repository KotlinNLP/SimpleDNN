/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multitasknetwork

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [MultiTaskNetwork]s which allows to allocate and release one when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a new network is created.
 *
 * @property model the model of the [MultiTaskNetwork]s of the pool
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @param inputMePropK the input layer k factor of the 'meProp' algorithm to propagate from the k (in percentage)
 *                     hidden nodes with the top errors (can be null)
 * @param outputMePropK a list of k factors (one for each output layer) of the 'meProp' algorithm to propagate from
 *                      the k (in percentage) output nodes with the top errors (the list and each element can be null)
 */
class MultitaskNetworksPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: MultiTaskNetworkModel,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean,
  private val inputMePropK: Double? = null,
  private val outputMePropK: List<Double?>? = null
) : ItemsPool<MultiTaskNetwork<InputNDArrayType>>() {

  /**
   * The factory of a new [MultiTaskNetwork].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [MultiTaskNetwork] with the given [id]
   */
  override fun itemFactory(id: Int) = MultiTaskNetwork<InputNDArrayType>(
    model = this.model,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput,
    inputMePropK = this.inputMePropK,
    outputMePropK = this.outputMePropK,
    id = id
  )
}
