/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class NeuralProcessorsPoolSpec : Spek({

  describe("a NeuralProcessorsPool") {

    on("getItem") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null),
        useDropout = false,
        propagateToInput = false)

      val processor1 = pool.getItem()
      val processor2 = pool.getItem()

      it("should return a processor with id 0 when called the first time") {
        assertEquals(0, processor1.id)
      }

      it("should return a processor with id 1 when called the second time") {
        assertEquals(1, processor2.id)
      }

      it("should contain the expected number of processors") {
        assertEquals(2, pool.size)
      }

      it("should have all the processors in use") {
        assertEquals(pool.size, pool.usage)
      }
    }

    on("releaseItem") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null),
        useDropout = false,
        propagateToInput = false)

      val processor1 = pool.getItem()
      pool.getItem()

      pool.releaseItem(processor1)

      it("should contain the expected number of processors") {
        assertEquals(2, pool.size)
      }

      it("should contain the expected number of processors in use") {
        assertEquals(1, pool.usage)
      }
    }

    on("releaseAll") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null),
        useDropout = false,
        propagateToInput = false)

      pool.getItem()
      pool.getItem()

      pool.releaseAll()

      it("should contain the expected number of processors") {
        assertEquals(2, pool.size)
      }

      it("should contain the expected number of processors in use") {
        assertEquals(0, pool.usage)
      }
    }

    on("getItem after releaseAll") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null),
        useDropout = false,
        propagateToInput = false)

      val processor1 = pool.getItem()
      val processor2 = pool.getItem()
      val processor3 = pool.getItem()
      val processor4 = pool.getItem()

      pool.releaseAll()

      it("should return the expected processor 1") {
        assertTrue { processor1 === pool.getItem() }
      }

      it("should return the expected processor 2") {
        assertTrue { processor2 === pool.getItem() }
      }

      it("should return the expected processor 3") {
        assertTrue { processor3 === pool.getItem() }
      }

      it("should return the expected processor 4") {
        assertTrue { processor4 === pool.getItem() }
      }
    }
  }
})
