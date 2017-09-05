/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.attentionnetwork.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class AttentionNetworkParamsErrorsAccumulatorSpec : Spek({

  describe("an AttentionNetworkParamsErrorsAccumulator") {

    on("initialization") {

      val accumulator = AttentionNetworkParamsErrorsAccumulator(inputSize = 2, attentionSize = 3, sparseInput = false)
      val errors = accumulator.getParamsErrors()

      it("should return zeros errors of the attention layer contextVector when calling getParamsErrors()") {

        val contextVectorErrors = errors.attentionParams.contextVector.values
        assertTrue {
          contextVectorErrors.equals(contextVectorErrors.zerosLike(), tolerance = 1.0e-08)
        }
      }

      it("should return zeros errors of the transform layer parameters when calling getParamsErrors()") {

        assertTrue {
          errors.transformParams.all { it as UpdatableDenseArray
            it.values.equals(it.values.zerosLike(), tolerance = 1.0e-08)
          }
        }
      }
    }

    on("accumulation") {

      it("should raise an Exception with params errors not compatible") {

        val accumulator = AttentionNetworkParamsErrorsAccumulator(inputSize = 2, attentionSize = 3, sparseInput = false)

        assertFails { accumulator.accumulate(paramsErrors = AttentionLayerUtils.buildAttentionNetworkParams1()) }
      }

      val accumulator = AttentionNetworkParamsErrorsAccumulator(inputSize = 4, attentionSize = 2, sparseInput = false)
      val errors1 = AttentionLayerUtils.buildAttentionNetworkParams1()
      val errors2 = AttentionLayerUtils.buildAttentionNetworkParams2()

      accumulator.accumulate(errors1)
      accumulator.accumulate(errors2)

      val accumulatedErrors = accumulator.getParamsErrors()
      val contextVectorErrors: DenseNDArray = accumulatedErrors.attentionParams.contextVector.values
      val w: DenseNDArray = accumulatedErrors.transformParams.unit.weights.values as DenseNDArray
      val b: DenseNDArray = accumulatedErrors.transformParams.unit.biases.values

      it("should match the expected accumulated context vector errors") {
        assertTrue {
          contextVectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.1)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected accumulated weights errors") {
        assertTrue {
          w.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(1.0, -0.4, 0.3, -0.8),
              doubleArrayOf(1.0, 0.5, -0.8, 0.4)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected accumulated biases errors") {
        assertTrue {
          b.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, -0.3)),
            tolerance = 1.0e-06
          )
        }
      }

      accumulator.reset()

      it("should contain zeros errors after calling reset()") {
        val resetErrors = accumulator.getParamsErrors()
        assertTrue {
          resetErrors.transformParams.all { it as UpdatableDenseArray
            it.values.equals(it.values.zerosLike(), tolerance = 1.0e-08)
          }
        }
      }
    }

    on("average") {

      val accumulator = AttentionNetworkParamsErrorsAccumulator(inputSize = 4, attentionSize = 2, sparseInput = false)
      val errors1 = AttentionLayerUtils.buildAttentionNetworkParams1()
      val errors2 = AttentionLayerUtils.buildAttentionNetworkParams2()

      accumulator.accumulate(errors1)
      accumulator.accumulate(errors2)
      accumulator.averageErrors()

      val accumulatedErrors = accumulator.getParamsErrors()
      val contextVectorErrors: DenseNDArray = accumulatedErrors.attentionParams.contextVector.values
      val w: DenseNDArray = accumulatedErrors.transformParams.unit.weights.values as DenseNDArray
      val b: DenseNDArray = accumulatedErrors.transformParams.unit.biases.values

      it("should match the expected average of the context vector errors") {
        assertTrue {
          contextVectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, -0.05)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected average of the weights errors") {
        assertTrue {
          w.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.5, -0.2, 0.15, -0.4),
              doubleArrayOf(0.5, 0.25, -0.4, 0.2)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected average of the biases errors") {
        assertTrue {
          b.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.15)),
            tolerance = 1.0e-06
          )
        }
      }
    }
  }
})
