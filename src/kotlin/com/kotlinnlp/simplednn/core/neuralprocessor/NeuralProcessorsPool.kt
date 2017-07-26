/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor

import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * A pool of [NeuralProcessor]s which allows to allocate and release processors when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a processor is created.
 *
 * @property neuralNetwork the [NeuralNetwork] which the processors of the pool will work with
 */
abstract class NeuralProcessorsPool<NeuralProcessorType: NeuralProcessor>(
  private val neuralNetwork: NeuralNetwork
) {

  /**
   * The pool of all the created processors.
   */
  private val pool = arrayListOf<NeuralProcessorType>()

  /**
   * A set containing the ids of processors not in use.
   */
  private val availableProcessors = mutableSetOf<Int>()

  /**
   * Get a processor currently not in use (setting it as in use).
   */
  fun getProcessor(): NeuralProcessorType {

    if (availableProcessors.size == 0) {
      this.addProcessor()
    }

    return this.popAvailableProcessor()
  }

  /**
   * Set a processor as available again.
   */
  fun releaseProcessor(processor: NeuralProcessorType) {
    this.availableProcessors.add(processor.id)
  }

  /**
   * Set all processors as available again.
   */
  fun releaseAll() {
    this.pool.forEach { this.availableProcessors.add(it.id) }
  }

  /**
   * Add a new processor to the pool.
   */
  private fun addProcessor() {

    val processor = this.processorFactory(id = this.pool.size)

    this.pool.add(processor)
    this.availableProcessors.add(processor.id)
  }

  /**
   * Pop the first available processor removing it from the list of available ones (the pool is required to be not
   * empty).
   *
   * @return the first available processor
   */
  private fun popAvailableProcessor(): NeuralProcessorType {

    val processorId: Int = this.availableProcessors.first()
    this.availableProcessors.remove(processorId)

    return this.pool[processorId]
  }

  /**
   * The factory of a new processor
   *
   * @param id the id of the processor to create
   *
   * @return a new [NeuralProcessor] with the given [id]
   */
  abstract fun processorFactory(id: Int): NeuralProcessorType
}
