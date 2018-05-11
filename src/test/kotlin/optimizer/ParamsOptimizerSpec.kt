/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package optimizer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.attention.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.attention.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.attentionnetwork.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class ParamsOptimizerSpec : Spek({

  describe("a ParamsOptimizer") {

    val learningRateMethod = LearningRateMethod(learningRate = 0.1)

    context("not compatible params errors") {

      val optimizer = ParamsOptimizer(
        params = AttentionNetworkParameters(inputSize = 2, attentionSize = 3, sparseInput = false),
        updateMethod = learningRateMethod
      )

      on("update") {

        it("should raise an Exception") {
          assertFailsWith<IllegalArgumentException> {
            optimizer.accumulate(paramsErrors = AttentionLayerUtils.buildAttentionNetworkParams1())
          }
        }
      }
    }

    context("compatible params errors") {

      val network = AttentionNetwork<DenseNDArray>(
        model = AttentionNetworkParameters(inputSize = 4, attentionSize = 2),
        inputType = LayerType.Input.Dense)

      val params = AttentionLayerUtils.buildAttentionNetworkParams1()
      network.model.attentionParams.contextVector.values.assignValues(params.attentionParams.contextVector.values)
      network.model.transformParams.zip(params.transformParams).forEach { (a, b) -> a.values.assignValues(b.values) }

      val optimizer = ParamsOptimizer(params = network.model, updateMethod = learningRateMethod)
      val errors1 = AttentionLayerUtils.buildAttentionNetworkParams1()
      val errors2 = AttentionLayerUtils.buildAttentionNetworkParams2()

      optimizer.accumulate(errors1)
      optimizer.accumulate(errors2)
      optimizer.update()

      val contextVector: DenseNDArray = network.model.attentionParams.contextVector.values
      val w: DenseNDArray = network.model.transformParams.unit.weights.values as DenseNDArray
      val b: DenseNDArray = network.model.transformParams.unit.biases.values as DenseNDArray

      it("should match the expected updated context vector") {
        assertTrue {
          contextVector.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.28, -0.495)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected updated weights") {
        assertTrue {
          w.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.25, 0.42, 0.185, -0.16),
              doubleArrayOf(0.15, -0.125, 0.14, 0.58)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected updated biases") {
        assertTrue {
          b.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.33, -0.385)),
            tolerance = 1.0e-06
          )
        }
      }
    }
  }
})
