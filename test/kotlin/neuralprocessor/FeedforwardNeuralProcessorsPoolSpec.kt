/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralprocessor

import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.SimpleRecurrentNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessorsPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class FeedforwardNeuralProcessorsPoolSpec : Spek({

  describe("a FeedforwardNeuralProcessorsPool") {

    on("getItem") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null))

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

    on("releaseItems") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null))

      val processor1 = pool.getItem()
      pool.getItem()

      pool.releaseItems(processor1)

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
          outputActivation = null))

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

    context("FeedforwardNeuralProcessorsPool") {

      val pool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = FeedforwardNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null))

      val processor = pool.getItem()

      it("should return a FeedforwardNeuralProcessor") {
        assertTrue { processor is FeedforwardNeuralProcessor<DenseNDArray> }
      }
    }

    context("RecurrentNeuralProcessorsPool") {

      val pool = RecurrentNeuralProcessorsPool<DenseNDArray>(
        neuralNetwork = SimpleRecurrentNeuralNetwork(
          inputSize = 5,
          hiddenSize = 3,
          hiddenActivation = null,
          outputSize = 2,
          outputActivation = null))

      val processor = pool.getItem()

      it("should return a RecurrentNeuralProcessor") {
        assertTrue { processor is RecurrentNeuralProcessor<DenseNDArray> }
      }
    }
  }
})
