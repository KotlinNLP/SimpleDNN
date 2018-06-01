/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.optimizer

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.attentionnetwork.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class ParamsErrorsAccumulatorSpec : Spek({

  describe("a ParamsErrorsAccumulator") {

    on("initialization") {

      val accumulator = ParamsErrorsAccumulator<NetworkParameters>()

      it("should raise an Exception when calling getParamsErrors() before accumulation") {
        assertFailsWith<IllegalArgumentException> { accumulator.getParamsErrors() }
      }
    }

    on("accumulation") {

      val accumulator = ParamsErrorsAccumulator<FeedforwardLayerParameters>()
      val errors1 = AttentionLayerUtils.buildTransformLayerParams1()
      val errors2 = AttentionLayerUtils.buildTransformLayerParams2()

      accumulator.accumulate(errors1)
      accumulator.accumulate(errors2)

      val accumulatedErrors = accumulator.getParamsErrors()
      val w: DenseNDArray = accumulatedErrors.unit.weights.values as DenseNDArray
      val b: DenseNDArray = accumulatedErrors.unit.biases.values as DenseNDArray

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

      it("should raise an error when calling getParamsErrors after a reset()") {
        assertFails { accumulator.getParamsErrors() }
      }
    }

    on("average") {

      val accumulator = ParamsErrorsAccumulator<FeedforwardLayerParameters>()
      val errors1 = AttentionLayerUtils.buildTransformLayerParams1()
      val errors2 = AttentionLayerUtils.buildTransformLayerParams2()

      accumulator.accumulate(errors1)
      accumulator.accumulate(errors2)
      accumulator.averageErrors()

      val accumulatedErrors = accumulator.getParamsErrors()
      val w: DenseNDArray = accumulatedErrors.unit.weights.values as DenseNDArray
      val b: DenseNDArray = accumulatedErrors.unit.biases.values as DenseNDArray

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
