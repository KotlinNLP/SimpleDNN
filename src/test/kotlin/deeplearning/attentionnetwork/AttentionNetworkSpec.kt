/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.attention.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class AttentionNetworkSpec : Spek({

  describe("an AttentionNetwork") {

    on("forward") {

      val network = AttentionNetwork<DenseNDArray>(
        inputType = LayerType.Input.Dense,
        model = AttentionLayerUtils.buildAttentionNetworkParams1())

      val output = network.forward(inputSequence = AttentionLayerUtils.buildInputSequence())

      it("should match the expected output") {
        assertTrue {
          output.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
        }
      }
    }

    on("forward with external attention arrays") {

      val network = AttentionNetwork<DenseNDArray>(
        inputType = LayerType.Input.Dense,
        model = AttentionLayerUtils.buildAttentionNetworkParams1())

      val inputSequence = AttentionLayerUtils.buildInputSequence()

      val output = network.forward(
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence))

      it("should match the expected output") {
        assertTrue {
          output.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
        }
      }
    }

    on("backward") {

      val network = AttentionNetwork<DenseNDArray>(
        inputType = LayerType.Input.Dense,
        model = AttentionLayerUtils.buildAttentionNetworkParams1())

      val errors = AttentionNetworkParameters(inputSize = 4, attentionSize = 2)

      network.forward(inputSequence = AttentionLayerUtils.buildInputSequence())
      network.backward(
        outputErrors = AttentionLayerUtils.buildOutputErrors(),
        paramsErrors = errors,
        propagateToInput = true)

      val inputErrors = network.getInputErrors()

      it("should match the expected errors of the first input") {
        assertTrue {
          inputErrors[0].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.076383, 0.148207, 0.021613, -0.175261)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          inputErrors[1].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.073648, 0.172493, 0.032455, -0.179081)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          inputErrors[2].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.048449, 0.181651, 0.046974, -0.147318)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
