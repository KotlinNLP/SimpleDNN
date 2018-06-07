/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.cfn

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.recurrent.cfn.CFNLayerParameters
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
class CFNLayerParametersSpec : Spek({

  describe("a CFNLayerParameters") {

    context("initialization") {

      on("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
          1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
          1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
          2.5, 2.6)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = CFNLayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val wIn = params.inputGate.weights.values
        val wFor = params.forgetGate.weights.values
        val wC = params.candidateWeights.values

        val bIn = params.inputGate.biases.values
        val bFor = params.forgetGate.biases.values

        val wInr = params.inputGate.recurrentWeights.values
        val wForr = params.forgetGate.recurrentWeights.values

        it("should contain the expected initialized weights of the input gate") {
          (0 until wIn.length).forEach { i -> assertEquals(initValues[i], wIn[i]) }
        }

        it("should contain the expected initialized weights of the forget gate") {
          (0 until wFor.length).forEach { i -> assertEquals(initValues[i + 6], wFor[i]) }
        }

        it("should contain the expected initialized weights of the candidate") {
          (0 until wC.length).forEach { i -> assertEquals(initValues[i + 12], wC[i]) }
        }

        it("should contain the expected initialized biases of the input gate") {
          (0 until bIn.length).forEach { i -> assertEquals(0.9, bIn[i]) }
        }

        it("should contain the expected initialized biases of the forget gate") {
          (0 until bFor.length).forEach { i -> assertEquals(0.9, bFor[i]) }
        }

        it("should contain the expected initialized recurrent weights of the input gate") {
          (0 until wInr.length).forEach { i -> assertEquals(initValues[i + 18], wInr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the forget gate") {
          (0 until wForr.length).forEach { i -> assertEquals(initValues[i + 22], wForr[i]) }
        }
      }

      on("sparse input") {

        val params = CFNLayerParameters(
          inputSize = 3,
          outputSize = 2,
          sparseInput = true,
          weightsInitializer = null,
          biasesInitializer = null)

        val wIn = params.inputGate.weights.values
        val wFor = params.forgetGate.weights.values
        val wC = params.candidateWeights.values

        it("should contain sparse weights of the input gate") {
          assertTrue { wIn is SparseNDArray }
        }

        it("should contain sparse weights of the forget gate") {
          assertTrue { wFor is SparseNDArray }
        }

        it("should contain sparse weights of the candidate") {
          assertTrue { wC is SparseNDArray }
        }

        it("should throw an Exception when trying to initialize") {
          assertFails {
            CFNLayerParameters(
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

      val params = CFNLayerParameters(inputSize = 3, outputSize = 2)

      val iterator = params.iterator()

      val wIn = params.inputGate.weights
      val wFor = params.forgetGate.weights
      val wC = params.candidateWeights

      val bIn = params.inputGate.biases
      val bFor = params.forgetGate.biases

      val wInr = params.inputGate.recurrentWeights
      val wForr = params.forgetGate.recurrentWeights

      on("iteration 1") {
        it("should return the weights of the input gate") {
          assertEquals(wIn, iterator.next())
        }
      }

      on("iteration 2") {
        it("should return the weights of the forget gate") {
          assertEquals(wFor, iterator.next())
        }
      }

      on("iteration 3") {
        it("should return the weights of the candidate") {
          assertEquals(wC, iterator.next())
        }
      }

      on("iteration 4") {
        it("should return the biases of the input gate") {
          assertEquals(bIn, iterator.next())
        }
      }

      on("iteration 5") {
        it("should return the biases of the forget gate") {
          assertEquals(bFor, iterator.next())
        }
      }

      on("iteration 6") {
        it("should return the recurrent weights of the input gate") {
          assertEquals(wInr, iterator.next())
        }
      }

      on("iteration 7") {
        it("should return the recurrent weights of the forget gate") {
          assertEquals(wForr, iterator.next())
        }
      }

      on("iteration 8") {
        it("should return true when calling hasNext()") {
          assertFalse(iterator.hasNext())
        }
      }
    }

    context("copy") {

      val params = CFNLayerParameters(inputSize = 3, outputSize = 2)
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
