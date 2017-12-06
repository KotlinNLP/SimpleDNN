/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.parameters

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.*

/**
 *
 */
class GRULayerParametersSpec : Spek({

  describe("a GRULayerParameters") {

    context("initialization") {

      on("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
          1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
          1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
          2.5, 2.6, 2.7, 2.8, 2.9, 3.0)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = GRULayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val wc = params.candidate.weights.values
        val wr = params.resetGate.weights.values
        val wp = params.partitionGate.weights.values

        val bc = params.candidate.biases.values
        val br = params.resetGate.biases.values
        val bp = params.partitionGate.biases.values

        val wcr = params.candidate.recurrentWeights.values
        val wrr = params.resetGate.recurrentWeights.values
        val wpr = params.partitionGate.recurrentWeights.values

        it("should contain the expected initialized weights of the candidate") {
          (0 until wc.length).forEach { i -> assertEquals(initValues[i], wc[i]) }
        }

        it("should contain the expected initialized weights of the reset gate") {
          (0 until wr.length).forEach { i -> assertEquals(initValues[i + 6], wr[i]) }
        }

        it("should contain the expected initialized weights of the partition gate") {
          (0 until wp.length).forEach { i -> assertEquals(initValues[i + 12], wp[i]) }
        }

        it("should contain the expected initialized biases of the candidate") {
          (0 until bc.length).forEach { i -> assertEquals(0.9, bc[i]) }
        }

        it("should contain the expected initialized biases of the reset gate") {
          (0 until br.length).forEach { i -> assertEquals(0.9, br[i]) }
        }

        it("should contain the expected initialized biases of the partition gate") {
          (0 until bp.length).forEach { i -> assertEquals(0.9, bp[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wcr.length).forEach { i -> assertEquals(initValues[i + 18], wcr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wrr.length).forEach { i -> assertEquals(initValues[i + 22], wrr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wpr.length).forEach { i -> assertEquals(initValues[i + 26], wpr[i]) }
        }
      }

      on("sparse input") {

        val params = GRULayerParameters(
          inputSize = 3,
          outputSize = 2,
          sparseInput = true,
          weightsInitializer = null,
          biasesInitializer = null)

        val wr = params.resetGate.weights.values
        val wp = params.partitionGate.weights.values
        val wc = params.candidate.weights.values

        it("should contain sparse weights of the reset gate") {
          assertTrue { wr is SparseNDArray }
        }

        it("should contain sparse weights of the partition gate") {
          assertTrue { wp is SparseNDArray }
        }

        it("should contain sparse weights of the candidate") {
          assertTrue { wc is SparseNDArray }
        }

        it("should throw an Exception when trying to initialize") {
          assertFails {
            GRULayerParameters(
              inputSize = 3,
              outputSize = 2,
              sparseInput = true,
              weightsInitializer = ConstantInitializer(0.1),
              biasesInitializer = ConstantInitializer(0.1))
          }
        }
      }
    }

    context("iteration") {

      val params = GRULayerParameters(inputSize = 3, outputSize = 2)

      val iterator = params.iterator()

      val wc = params.candidate.weights
      val wr = params.resetGate.weights
      val wp = params.partitionGate.weights

      val bc = params.candidate.biases
      val br = params.resetGate.biases
      val bp = params.partitionGate.biases

      val wcr = params.candidate.recurrentWeights
      val wrr = params.resetGate.recurrentWeights
      val wpr = params.partitionGate.recurrentWeights

      on("iteration 1") {
        it("should return the weights of the candidate") {
          assertEquals(wc, iterator.next())
        }
      }

      on("iteration 2") {
        it("should return the weights of the reset gate") {
          assertEquals(wr, iterator.next())
        }
      }

      on("iteration 3") {
        it("should return the weights of the partition gate") {
          assertEquals(wp, iterator.next())
        }
      }

      on("iteration 4") {
        it("should return the biases of the candidate") {
          assertEquals(bc, iterator.next())
        }
      }

      on("iteration 5") {
        it("should return the biases of the reset gate") {
          assertEquals(br, iterator.next())
        }
      }

      on("iteration 6") {
        it("should return the biases of the partition gate") {
          assertEquals(bp, iterator.next())
        }
      }

      on("iteration 7") {
        it("should return the recurrent weights of the candidate") {
          assertEquals(wcr, iterator.next())
        }
      }

      on("iteration 8") {
        it("should return the recurrent weights of the reset gate") {
          assertEquals(wrr, iterator.next())
        }
      }

      on("iteration 9") {
        it("should return the recurrent weights of the partition gate") {
          assertEquals(wpr, iterator.next())
        }
      }

      on("iteration 10") {
        it("should return true when calling hasNext()") {
          assertFalse(iterator.hasNext())
        }
      }
    }

    context("copy") {

      val params = GRULayerParameters(inputSize = 3, outputSize = 2)
      val clonedParams = params.copy()

      it("should return a new element") {
        assertNotEquals(params, clonedParams)
      }

      it("should return params with same values") {
        assertTrue {
          params.zip(clonedParams).all { it.first.values.equals(it.second.values, tolerance = 1.0e-06) }
        }
      }
    }
  }
})
